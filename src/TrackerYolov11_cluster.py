# -*- coding: utf-8 -*-
'''
Title: Tracker (Headless / Excel Column Lists / Auto-Video) - OPTIMIZED BATCH GPU
Description: 1. Reads metadata from *RecordingMeta.xlsx.
             2. Optimized for Batch Processing (Maximizes GPU Usage).
             3. Multi-GPU support via --gpu_id.
'''

from itertools import groupby
from datetime import date, timedelta, datetime
from pathlib import Path
from collections import deque
from tools import mask
import cv2
from ultralytics import YOLO 
import os
import math
import time
import logging
import threading
import queue 
import numpy as np
import pandas as pd
import sys
import argparse
import glob
from tqdm import tqdm
import torch

# --- CONFIGURATION ---
FONT = cv2.FONT_HERSHEY_TRIPLEX
font = cv2.FONT_HERSHEY_PLAIN 
colors = np.random.uniform(0, 255, size=(100, 3))

def points_dist(p1, p2):
    dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return dist

def convert_milli(time):
    sec = (time / 1000) % 60
    minute = (time / (1000 * 60)) % 60
    hr = (time / (1000 * 60 * 60)) % 24
    return f'{int(hr):02d}:{int(minute):02d}:{sec:.3f}'

def safe_int_str(val):
    try:
        if pd.isna(val): return ""
        return str(int(float(val)))
    except:
        return str(val)

# --- CLASS: Threaded Video Writer ---
class ThreadedVideoWriter:
    def __init__(self, path, fourcc, fps, frame_size):
        self.output_file = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self.queue = queue.Queue()
        self.stopped = False
        self.thread = threading.Thread(target=self.write_frames, daemon=True)
        self.thread.start()

    def write(self, frame):
        if not self.stopped:
            self.queue.put(frame)

    def write_frames(self):
        while True:
            if self.stopped and self.queue.empty():
                break
            try:
                frame = self.queue.get(timeout=1) 
                self.output_file.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

    def release(self):
        self.stopped = True
        self.thread.join()
        self.output_file.release()

# --- CLASS: Tracker ---
class Tracker:
    def __init__(self, vp, nl, out, metadata, onnx_weight, gpu_id=0, batch_size=32):
        '''Tracker class initialisations'''
        self.metadata = metadata 
        self.out_path = out 
        self.model_path = onnx_weight 
        self.gpu_id = gpu_id 
        self.batch_size = batch_size # New Batch Size Parameter
        
        threads = list()
        
        self.load_network(self.model_path)

        session = threading.Thread(target=self.load_session, args=(vp, nl, 1, out))
        threads.append(session)
        session.start()
        session.join()
            
        print('\n -Network loaded- ')

        print("Caching node dictionary...")
        self.nodes_dict = mask.create_node_dict(self.node_list)

        self.start_nodes_locations = self.find_location(self.start_nodes, self.goal)
        print('\n  ________  SUMMARY SESSION  ________  ')
        print(f'Batch Size: {self.batch_size}')
        print(f'GPU ID: {self.gpu_id}')
        print('\nTotal trials current session:', self.num_trials, '\n\nGoal location node ', self.goal)
        
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        logfile_name = '{}/log_{}_{}.log'.format(out, str(self.date), 'Rat' + self.rat)
        
        if not os.path.exists(out):
            os.makedirs(out, exist_ok=True)

        fh = logging.FileHandler(str(logfile_name))
        formatter = logging.Formatter('%(levelname)s : %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.ts_file_loaded = False
        try:
            specific_ts_path = os.path.join(out, f'{str(self.date)}_Rat{str(self.rat)}_framewise_ts.csv')
            if os.path.exists(specific_ts_path):
                 self.sync_ts_dict = pd.read_csv(specific_ts_path, index_col=0).to_dict()
                 self.ts_file_loaded = True
            else:
                 stitched_ts_path = os.path.join(out, 'stitched_framewise_ts.csv')
                 if os.path.exists(stitched_ts_path):
                     self.sync_ts_dict = pd.read_csv(stitched_ts_path, index_col=0).to_dict()
                     self.ts_file_loaded = True
                 else:
                     raise FileNotFoundError
        except Exception:
             self.sync_ts_dict = {"Corrected Time Stamp": {}} 

        self.frame_data_log = []

        self.run_vid()
    
    def change_name_csv(self, output_path):
        csvfile_name = os.path.join(output_path,f'{str(self.date)}_Rat{str(self.rat)}_framewise_ts.csv')
        stitched_name = os.path.join(output_path,'stitched_framewise_ts.csv')
        
        if os.path.exists(stitched_name):
            try:
                if os.path.exists(csvfile_name):
                    os.remove(csvfile_name)
                os.rename(stitched_name, csvfile_name)
            except OSError as e:
                print(f"Error renaming file: {e}")
        else:
            pass
        
    def load_network(self, model_path):
        print(f"Loading YOLO11 model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            
            if torch.cuda.is_available():
                if self.gpu_id < torch.cuda.device_count():
                    self.device = self.gpu_id
                    print(f" >> SUCCESS: Using GPU ID {self.device}: {torch.cuda.get_device_name(self.device)}")
                else:
                    print(f" >> WARNING: Requested GPU ID {self.gpu_id} not found. Defaulting to GPU 0.")
                    self.device = 0
            else:
                print(" >> WARNING: No GPU detected. Running on CPU.")
                self.device = 'cpu'

            print("YOLO11 Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def load_session(self, vp, nl, n, out):
        self.start_point = self.metadata['start_point']
        self.custom_trial = self.metadata['custom_trial']
        self.rat = self.metadata['rat']
        self.date = self.metadata['date']
        self.num_trials = self.metadata['num_trials']
        self.goal = self.metadata['goal']
        self.trial_type = self.metadata['trial_type']
        self.start_nodes = self.metadata['start_nodes_list']
        self.special_trials = self.metadata['special_trials_list']
        self.repeat = self.metadata['repeat']
        self.day_num = self.metadata['day']
        self.session_num = self.metadata['session']
        self.unnormal_intervals = self.metadata.get('unnormal_intervals', {})

        self.node_list = str(nl)
        self.cap = cv2.VideoCapture(str(vp))
        self.start_trial = True 
        self.end_session = False 
        self.check = False 
        self.record_detections = False 
        self.goal_location = None
        self.reached = False
        self.pos_centroid = None 
        self.center_researcher = None
        
        self.last_rat_pos = None         
        self.last_researcher_pos = None 

        if self.start_point is None:
           self.trial_num = 1
        else:
           self.trial_num = int(self.custom_trial)
        self.counter = 0 
        self.count_rat = 0
        self.count_head = 0
        self.start_time = 0 
        
        self.normal_trial = False
        self.NGL = False
        self.probe = False

        self.goal_residence_timer = 0.0
        self.centroid_list = deque(maxlen=500)
        self.node_pos = []
        self.time_points = []
        self.node_id = [] 
        self.saved_nodes = []
        self.saved_velocities = []
        self.summary_trial = []
        self.store_fps = [] 
        self.locked_to_head = False   
        self.start_node_center = None
        self.covering_start_node = False
        self.cover_required_time = 1
        self.start_node_radius = 20
        self.goal_node_radius = 25
        self.save = '{}/{}_{}'.format(out, str(self.date), 'Rat' + self.rat + '.txt') 
        
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.save_video = '{}/{}_{}.mp4'.format(out, str(self.date), 'Rat' + self.rat) 
        self.vid_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.out = ThreadedVideoWriter('{}'.format(self.save_video), self.codec, self.vid_fps, (1176, 712))
        self.researcher_goal_timer = 0.0

    # --- OPTIMIZED BATCH RUNNER ---
    # --- OPTIMIZED BATCH RUNNER (FIXED LOGGING) ---
    def run_vid(self):
        print(f'\nStarting video processing (Batch Mode: {self.batch_size}).....\n')
        
        if self.start_point is None:
            with open(self.save, 'a+') as file:
                file.write(f"Rat number: {self.rat} , Date: {self.date} \n")
        self.Start_Time = time.time()
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        
        if self.start_point is not None:
            frame_index = int(float(self.start_point) * self.vid_fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        pbar = tqdm(total=total_frames - frame_index, unit='frames', desc='Processing', ncols=100)

        # Batch Buffers
        batch_frames_resized = []
        batch_original_times = [] # Store timestamps/metadata
        
        while True:
            # 1. Fill Batch
            current_batch_size = 0
            while current_batch_size < self.batch_size:
                success, frame = self.cap.read()
                if not success:
                    break
                
                # Pre-resize (CPU) - YOLO expects the size you want to process
                resized_frame = cv2.resize(frame, (1176, 712))
                
                frame_time_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                
                batch_frames_resized.append(resized_frame)
                batch_original_times.append(frame_time_msec)
                
                current_batch_size += 1

            # Check if we have anything to process
            if not batch_frames_resized:
                if not self.end_session:
                    self.calculate_velocity(self.time_points)
                    self.save_to_file(self.save)
                break # EOF

            # 2. Batch Inference (GPU)
            results = self.model(batch_frames_resized, verbose=False, conf=0.3, iou=0.8, device=self.device)

            # 3. Process Results Sequentially
            for i, result in enumerate(results):
                self.frame_time = batch_original_times[i]
                self.converted_time = convert_milli(int(self.frame_time))
                self.disp_frame = batch_frames_resized[i] # The frame we will draw on
                
                # Logic Processing
                self.t1 = time.time() # FPS calculation
                
                # Logic Update
                self.process_result(result) 
                
                # Visualization
                self.annotate_frame(self.disp_frame)
                self.out.write(self.disp_frame)
                
                # CSV logging (Memory list)
                self.log_frame_data(frame_index)

                # === [FIX START] RESTORED LOGGING LOGIC ===
                # 这部分逻辑在之前优化 batch 时丢失了，导致 .log 文件为空
                if self.record_detections:
                    ts_val = self.sync_ts_dict.get("Corrected Time Stamp", {}).get(frame_index, "N/A")
                    if self.saved_nodes:
                        self.logger.info(
                            f'{self.converted_time} {ts_val} : The rat position is: {self.pos_centroid} @ {self.saved_nodes[-1]}')
                    else:
                        self.logger.info(
                            f'{self.converted_time} {ts_val} : The rat position is: {self.pos_centroid}')
                # === [FIX END] ===

                frame_index += 1
                pbar.update(1)

                if self.end_session:
                    break
            
            # Clear buffers for next batch
            batch_frames_resized = []
            batch_original_times = []
            
            if self.end_session:
                break

        pbar.close()
        self.export_tracking_data()
        
        # Cleanup
        end = time.time()
        hours, rem = divmod(end - self.Start_Time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\nTracking process finished in: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        self.cap.release()
        self.out.release()

    def log_frame_data(self, frame_itr):
        rat_x = self.pos_centroid[0] if self.pos_centroid else np.nan
        rat_y = self.pos_centroid[1] if self.pos_centroid else np.nan
        res_x = self.Researcher[0] if self.Researcher else np.nan
        res_y = self.Researcher[1] if self.Researcher else np.nan

        trial_num = self.trial_num if self.record_detections else np.nan
        
        self.frame_data_log.append({
            'Frame_Index': frame_itr,
            'Trial_Num': trial_num,
            'Rat_X': rat_x, 'Rat_Y': rat_y,
            'Researcher_X': res_x, 'Researcher_Y': res_y,
            'JP_S_X': np.nan, 'JP_S_Y': np.nan, 'JP_L_X': np.nan, 'JP_L_Y': np.nan
        })

    # --- REPLACED CNN FUNCTION ---
    # Now accepts a 'result' object from the batch, not a raw frame
    def process_result(self, result):
        self.Rat = None
        self.Researcher = None
        
        rat_candidates = []
        researcher_candidates = []

        detected_head_this_frame = False
        detected_rat_body_this_frame = False

        # Iterate Detections from Result Object
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            confidence = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = self.model.names[cls_id]
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            centroid = (center_x, center_y)

            # Draw
            color = colors[cls_id % len(colors)]
            cv2.rectangle(self.disp_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.disp_frame, f"{label} {confidence:.2f}", (x1, y1 + 20), font, 1, (255, 255, 255), 1)

            if label == 'head':
                rat_candidates.append((confidence, centroid, 'head'))
                detected_head_this_frame = True
            elif label == 'rat':
                rat_candidates.append((confidence, centroid, 'rat'))
                detected_rat_body_this_frame = True
            elif label == 'researcher':
                researcher_candidates.append((confidence, centroid))

        # Selection Logic
        if rat_candidates:
            rat_candidates.sort(key=lambda x: x[0], reverse=True)
            best_conf, best_centroid, best_label = rat_candidates[0]
            if best_label == 'head': self.locked_to_head = True
            self.Rat = best_centroid

        if researcher_candidates:
            researcher_candidates.sort(key=lambda x: x[0], reverse=True)
            self.Researcher = researcher_candidates[0][1]

        # Post-Detection State Machine (Updates self.pos_centroid, etc.)
        self.update_tracker_logic(detected_head_this_frame, detected_rat_body_this_frame)

    # --- MOVED STATE MACHINE LOGIC HERE ---
    def update_tracker_logic(self, detected_head_this_frame, detected_rat_body_this_frame):
        if self.Rat is not None: self.last_rat_pos = self.Rat
        if self.Researcher is not None: self.last_researcher_pos = self.Researcher

        active_rat_pos = self.Rat if self.Rat is not None else self.last_rat_pos
        
        # Unnormal Trial Check
        if not self.start_trial and not self.end_session and self.trial_num in self.unnormal_intervals:
            _, end_block_abs = self.unnormal_intervals[self.trial_num]
            current_abs_minutes = (self.frame_time / (1000 * 60)) % 60
            if current_abs_minutes >= end_block_abs:
                self.force_end_trial()
                return

        # Start Trigger (Researcher near Rat)
        if self.Researcher and active_rat_pos and not self.record_detections:
            if (not self.start_trial and not self.end_session and 
                not self.record_detections and points_dist(active_rat_pos, self.Researcher) <= 800): 
                self.start_trial = True
                self.trial_num += 1
                self.check = False

        if active_rat_pos:
            if self.start_trial:
                self.find_start(active_rat_pos)
            
            if self.record_detections:
                if detected_head_this_frame: self.count_head += 1
                elif detected_rat_body_this_frame: self.count_rat += 1
                
                self.object_detection(rat=active_rat_pos)
                
                # Check Researcher vs Goal (During trial)
                if self.Researcher is not None and self.goal_location is not None:
                    if points_dist(self.Researcher, self.goal_location) <= 50:
                        self.researcher_goal_timer += (1.0 / self.vid_fps)
                        if self.researcher_goal_timer >= 10.0:
                            self.force_end_trial()
                    else:
                        self.researcher_goal_timer = 0.0

        # Check Researcher vs Goal (General)
        if self.Researcher is not None and self.goal_location is not None:
            if points_dist(self.Researcher, self.goal_location) <= 80:
                allow_end = True
                if self.probe:
                    cur = (self.frame_time / (1000 * 60)) % 60
                    if (cur - self.start_time) < 2.0: allow_end = False
                if self.check_immunity(): allow_end = False
                    
                if allow_end:
                    self.researcher_goal_timer += (1.0 / self.vid_fps)
                    if self.researcher_goal_timer >= 10.0:
                        self.force_end_trial()
                else:
                    self.researcher_goal_timer = 0.0

        # Researcher Covering Start Node Logic
        researcher_covers_start = False
        if (not self.start_trial and not self.record_detections and 
            not self.end_session and self.counter < len(self.start_nodes_locations)):
            
            self.start_node_center = self.start_nodes_locations[self.counter]
            if self.Researcher is not None: 
                if points_dist(self.Researcher, self.start_node_center) <= 40:
                    researcher_covers_start = True

            if researcher_covers_start:
                if not self.covering_start_node:
                    self.covering_start_node = True
                    self.cover_start_timer = 0.0
                self.cover_start_timer += self.frame_time
                if self.cover_start_timer >= self.cover_required_time:
                    self.start_trial = True
                    self.trial_num += 1
                    self.check = False
                    self.covering_start_node = False
                    self.cover_start_timer = 0.0
            else:
                if self.covering_start_node:
                    self.covering_start_node = False
                    self.cover_start_timer = 0.0

    def force_end_trial(self):
        self.normal_trial = False
        self.NGL = False
        self.probe = False
        self.reached = False
        self.end_trial()
        if self.trial_num in self.unnormal_intervals: # Special handling for unnormal
             self.start_trial = True  
             self.trial_num += 1      
             self.check = False   
        else:
             self.researcher_goal_timer = 0.0

    def export_tracking_data(self):
        print("\n>> Compiling tracking data to CSV...")
        df_tracking = pd.DataFrame(self.frame_data_log)
        
        if not df_tracking.empty:
            df_tracking['Frame_Index'] = df_tracking['Frame_Index'].astype(int)

        if self.ts_file_loaded:
            try:
                raw_ts_data = self.sync_ts_dict.get('Corrected Time Stamp', self.sync_ts_dict)
                df_master = pd.DataFrame.from_dict(raw_ts_data, orient='index', columns=['Timestamp'])
                df_master.index.name = 'Frame_Index'
                df_master.index = df_master.index.astype(int)
                df_master.sort_index(inplace=True)

                df_final = pd.merge(df_master, df_tracking, on='Frame_Index', how='left')
                if 'Timestamp_y' in df_final.columns:
                    df_final.rename(columns={'Timestamp_x': 'Timestamp'}, inplace=True)
                    df_final.drop(columns=['Timestamp_y'], inplace=True)
                df_tracking = df_final
            except Exception as e:
                print(f"Warning: Merge failed. {e}")
        
        filename = f"{self.date}_Rat{self.rat}_Coordinates_Full.csv"
        save_path = os.path.join(self.out_path, filename)
        df_tracking.to_csv(save_path, index=False)
        print(f">> Saved: {save_path}")

    def find_start(self, center_rat):
        node = self.start_nodes_locations[self.counter]
        self.locked_to_head = False 
        if points_dist(center_rat, node) < 60:
            self.logger.info('Recording Trial {}'.format(self.trial_num))
            
            if self.trial_num == 1 and int(self.trial_type) != 1:
                self.start_time = (self.frame_time / (1000 * 60)) % 60
                if int(self.trial_type) == 3: self.probe = True
                if int(self.trial_type) == 2: self.NGL = True
            if int(self.trial_type) == 4:
                for n in self.special_trials:
                    if int(n) == self.trial_num:
                        self.NGL = True
                        self.start_time = (self.frame_time / (1000 * 60)) % 60
            if not self.probe and not self.NGL: self.normal_trial = True

            self.node_pos = []
            self.centroid_list = []
            self.time_points = []
            self.summary_trial = []
            self.saved_nodes = []
            self.node_id = [] 
            self.saved_velocities = []
            self.record_detections = True  
            self.researcher_goal_timer = 0.0
            self.pos_centroid = node
            self.centroid_list.append(self.pos_centroid)
            self.start_trial = False  
            
    def check_immunity(self):
        if self.trial_num in self.unnormal_intervals:
            start_block, end_block = self.unnormal_intervals[self.trial_num]
            current_abs_minutes = (self.frame_time / (1000 * 60)) % 60
            if start_block <= current_abs_minutes <= end_block: return True
        return False
    
    def object_detection(self, rat):
        self.pos_centroid = rat
        self.centroid_list.append(self.pos_centroid)
        is_immune = self.check_immunity()

        if self.NGL:
            minutes = self.timer(start=self.start_time)
            if not self.reached:
                if points_dist(self.pos_centroid, self.goal_location) <= 20: self.reached = True
            if minutes >= 10:
                print('\n\n >>> Ten minute passed... Goal location reached:', self.reached)
                if self.reached and not is_immune:
                    self.force_end_trial()

        if self.probe:
            minutes = self.timer(start=self.start_time)
            if minutes >= 2:
                if points_dist(self.pos_centroid, self.goal_location) <= self.goal_node_radius:
                    if not is_immune: self.force_end_trial()

        if self.normal_trial:
            if points_dist(self.pos_centroid, self.goal_location) <= self.goal_node_radius:
                if not is_immune: self.force_end_trial()
    
    def end_trial(self):
        self.pos_centroid = self.goal_location
        self.centroid_list.append(self.pos_centroid)
        # We don't annotate here, already done in loop
        if self.saved_nodes:
            self.logger.info(f'{self.converted_time} : Rat @ {self.pos_centroid} ({self.saved_nodes[-1]})')
        else:
            self.logger.info(f'{self.converted_time} : Rat @ {self.pos_centroid}') 
        self.calculate_velocity(self.time_points)
        self.save_to_file(self.save)

        self.counter += 1 
        if self.counter == int(self.num_trials):
            self.end_session = True
        self.record_detections = False
        self.count_rat = 0
        self.count_head = 0

    def timer(self, start):
        end = (self.frame_time / (1000 * 60)) % 60
        duration = end - start
        if duration < 0: duration += 60
        return int(duration)

    def calculate_velocity(self, time_points):
        bridges = {('124', '201'): 0.60, ('121', '302'): 1.72,
                   ('223', '404'): 1.69, ('324', '401'): 0.60, ('305', '220'): 0.60}
        if len(time_points) > 2:
            lenght = 0
            speed = 0
            format = '%H:%M:%S.%f'
            for i in range(0, len(time_points)):
                start_node = time_points[i][1]
                start_time = datetime.strptime((time_points[i][0]), format).time()
                j = i + 1
                if j == len(time_points):
                    self.last_node = time_points[i][1]
                else:
                    end_node = time_points[j][1]
                    end_time = datetime.strptime((time_points[j][0]), format).time()
                    diff = timedelta(hours=end_time.hour - start_time.hour,
                                     minutes=end_time.minute - start_time.minute,
                                     seconds=end_time.second - start_time.second,
                                     microseconds=end_time.microsecond - start_time.microsecond).total_seconds()
                    
                    if (start_node, end_node) in bridges: lenght = bridges[(start_node, end_node)]
                    elif (end_node, start_node) in bridges: lenght = bridges[(end_node, start_node)]
                    else: lenght = 0.30 
                    
                    try: speed = round(float(lenght) / float(diff), 3)
                    except ZeroDivisionError: speed = 0
                    finally:
                        self.summary_trial.append([(start_node, end_node), (time_points[i][0], time_points[j][0]), diff, lenght, speed])
                        self.saved_velocities.append(speed)

    @staticmethod
    def annotate_node(frame, point, node, t):
        if t == 1:
            cv2.circle(frame, point, 20, (0, 255, 0), 2)
            # FIXED: FONT comes before 0.5
            cv2.putText(frame, str(node), (point[0]-16, point[1]), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Start', (point[0]-16, point[1]-22), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if t == 2:
            cv2.circle(frame, point, 20, (20, 110, 245), 1)
            # FIXED: FONT comes before 0.5
            cv2.putText(frame, str(node), (point[0]-16, point[1]), FONT, 0.5, (0, 69, 255), 1, cv2.LINE_AA)
        if t == 3:
            cv2.circle(frame, point, 20, (0, 0, 250), 2)
            # FIXED: FONT comes before 0.5
            cv2.putText(frame, str(node), (point[0]-16, point[1]), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'End', (point[0]-16, point[1]-22), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    def annotate_frame(self, frame):
        nodes_dict = self.nodes_dict 
        
        cv2.putText(frame, str(self.converted_time), (970, 670), FONT, 0.75, (240, 240, 240), 1)
        fps = 1. / (time.time() - self.t1) 
        self.store_fps.append(fps)
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (970, 650), FONT, 0.75, (240, 240, 240), 1)
        self.annotate_node(frame, point=self.goal_location, node=self.goal, t=3)
        
        if self.start_trial and self.counter < len(self.start_nodes):
            cv2.putText(frame, f'Next trial: {self.trial_num}', (60, 60), FONT, 0.75, (255, 255, 255), 1)
            cv2.putText(frame, 'Waiting start new trial...', (60, 80), FONT, 0.75, (255, 255, 255), 1)
            start_pos = self.start_nodes_locations[self.counter]
            self.annotate_node(frame, point=start_pos, node=self.start_nodes[self.counter], t=1)

        if self.record_detections:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) <= 20:
                    self.saved_nodes.append(node_name)
                    self.node_pos.append(nodes_dict[node_name])
                    if len(self.time_points) == 0:
                        self.time_points.append([self.converted_time, node_name])
                    if node_name != self.saved_nodes[(len(self.saved_nodes)) - 2]:
                        self.time_points.append([self.converted_time, node_name])

            cv2.putText(frame, 'Trial:' + str(self.trial_num), (60, 60), FONT, 0.75, (255, 255, 255), 1)
            cv2.putText(frame, 'Currently writing to file...', (60, 80), FONT, 0.75, (255, 255, 255), 1)
            cv2.putText(frame, "Rat Count: " + str(self.count_rat), (40, 130), FONT, 0.65, (255, 255, 255), 1)
            cv2.putText(frame, "Rat-head Count: " + str(self.count_head), (40, 160), FONT, 0.65, (255, 255, 255), 1)

            if len(self.centroid_list) >= 2:
                for i in range(1, len(self.centroid_list)):
                    cv2.line(frame, self.centroid_list[i], self.centroid_list[i - 1], (255, 0, 60), 1)
            cv2.line(frame, (self.pos_centroid[0] - 5, self.pos_centroid[1]), (self.pos_centroid[0] + 5, self.pos_centroid[1]), (0, 255, 0), 2)
            cv2.line(frame, (self.pos_centroid[0], self.pos_centroid[1] - 5), (self.pos_centroid[0], self.pos_centroid[1] + 5), (0, 255, 0), 2)

            start_index = max(0, len(self.saved_nodes) - 50)
            for i in range(start_index, len(self.saved_nodes)):
                self.annotate_node(frame, point=self.node_pos[i], node=self.saved_nodes[i], t=2)

    def save_to_file(self, fname):
        savelist = []
        with open(fname, 'a+') as file:
            for k, g in groupby(self.saved_nodes): savelist.append(k)
            file.writelines('%s,' % items for items in savelist)
            file.write('\nSummary Trial {}\nStart-Next Nodes// Time points(s) //Seconds//Lenght(cm)// Velocity(m/s)\n'.format(self.trial_num))
            for i in range(0, len(self.summary_trial)):
                line = " ".join(map(str, self.summary_trial[i]))
                file.write(line + '\n')
            file.write('\n')
        file.close()

    def find_location(self, start_nodes, goal):
        nodes_dict = self.nodes_dict
        start_nodes_locations = []
        for node_name in nodes_dict:
            if node_name == str(goal): self.goal_location = nodes_dict[node_name]
        for node in start_nodes:
            for node_name in nodes_dict:
                if node_name == str(node): start_nodes_locations.append(nodes_dict[node_name])
        return start_nodes_locations

def parse_metadata_xlsx(xlsx_path):
    print(f"Reading configuration from: {xlsx_path}")
    try:
        df = pd.read_excel(xlsx_path, engine='openpyxl')
        row0 = df.iloc[0] 
        trial_type = "1"
        if 'Trial_Type' in row0 and not pd.isna(row0['Trial_Type']): trial_type = safe_int_str(row0['Trial_Type'])
             
        start_pt = None
        s_min = float(row0.get('Start_Min', 0))
        s_sec = float(row0.get('Start_Sec', 0))
        if s_min > 0 or s_sec > 0: start_pt = (s_min * 60) + s_sec

        s_nodes = []
        if 'Start_Nodes' in df.columns: s_nodes = df['Start_Nodes'].dropna().astype(int).tolist()

        sp_trials = []
        if 'Special_Trials' in df.columns: sp_trials = df['Special_Trials'].dropna().astype(int).tolist()

        un_dict = {}
        if 'Unnormal_Intervals' in df.columns:
            un_list = df['Unnormal_Intervals'].dropna().astype(str).tolist()
            for item in un_list:
                item = item.strip()
                if ":" in item and "-" in item:
                    parts = item.split(":")
                    try:
                        t_num = int(float(parts[0]))
                        times = parts[1].split("-")
                        un_dict[t_num] = (float(times[0]), float(times[1]))
                    except ValueError: pass

        metadata = {
            'start_point': start_pt,
            'custom_trial': int(float(row0.get('Start_At_Trial_Num', 1))),
            'rat': safe_int_str(row0['Rat_ID']),
            'date': safe_int_str(row0['Date']),
            'repeat': safe_int_str(row0['Repeat']),
            'day': safe_int_str(row0['Day']),
            'session': safe_int_str(row0['Session']),
            'num_trials': safe_int_str(row0['Num_Trials']),
            'goal': safe_int_str(row0['Goal_Node']),
            'prev_goal': safe_int_str(row0.get('Prev_Goal_Node', 'N/A')),
            'trial_type': trial_type,
            'start_nodes_list': s_nodes,
            'special_trials_list': sp_trials,
            'unnormal_intervals': un_dict
        }
        return metadata

    except Exception as e:
        print(f"Error parsing Excel file: {e}")
        raise e

# --- MAIN ---
if __name__ == "__main__":
    try:
        node_list = Path('src/tools/node_list_new.csv').resolve()
        print('\n\nTracker version: v3.0-Batch-GPU (Headless / Mass Analysis)\n\n')

        parser = argparse.ArgumentParser(description="Tracker Headless Mode")
        parser.add_argument('--input_folder', required=True, help="Folder containing 'stitched.mp4'")
        parser.add_argument('--output_folder', required=True, help="Path to output directory")
        parser.add_argument('--onnx_weight', required=True, help="Path to .pt model")
        parser.add_argument('--gpu_id', type=int, default=0, help="ID of the GPU to use (0, 1, etc.)")
        parser.add_argument('--batch_size', type=int, default=32, help="Frames per GPU batch (Try 16 or 32 for powerful GPUs)")
        
        args = parser.parse_args()
        
        in_p = args.input_folder
        out_p = args.output_folder
        model_path = args.onnx_weight 
        
        vid_p = os.path.join(in_p, 'stitched.mp4')
        if not os.path.exists(vid_p):
            print(f"ERROR: Video file not found at: {vid_p}")
            sys.exit(1)

        meta_files = glob.glob(os.path.join(in_p, '*RecordingMeta.xlsx'))
        if not meta_files:
            print(f"ERROR: No *RecordingMeta.xlsx found in: {in_p}")
            sys.exit(1)
            
        xlsx_file = meta_files[0] 
        metadata = parse_metadata_xlsx(xlsx_file)

        tracker = Tracker(vp=vid_p, nl=node_list, out=out_p, metadata=metadata, 
                          onnx_weight=model_path, gpu_id=args.gpu_id, batch_size=args.batch_size)
        
        tracker.change_name_csv(out_p)
        print("Done.")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\nProcess interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)