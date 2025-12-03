import numpy as np
import cv2
import csv

#reading the node lists as dictionary data type
def create_node_dict(node_list):
    nodes_dict = {}
    with open(node_list, 'r') as nl:
        read_nodes = csv.reader(nl)
        for node_list_values in read_nodes:
            x = int(node_list_values[1])
            y = int(node_list_values[2])
            point = (x , y)
        
            nodes_dict.update({node_list_values[0] : point})

    return nodes_dict


#mask for the hex maze(vid size - [1176, 712])
def create_mask(node_list):
    
    black_im =  np.zeros(shape=[712 , 1176, 3], dtype = np.uint8)
    
    #mask initialisations
    flower_graph = {1: [2, 7],
                2: [1, 3],
                3: [2, 4, 9],
                4: [3, 5],
                5: [4, 11],
                6: [7, 13],
                7: [6, 1, 8],
                8: [7, 9, 15],
                9: [3, 8, 10],
                10: [9, 11, 17],
                11: [5, 10, 12],
                12: [11, 19],
                13: [6, 14],
                14: [13, 15, 20],
                15: [8, 14, 16],
                16: [15, 17, 22],
                17: [10, 16, 18],
                18: [17, 19, 24],
                19: [12, 18],
                20: [14, 21],
                21: [20, 22],
                22: [16, 21, 23],
                23: [22, 24],
                24: [18, 23]}
    
    island_prefixes = ['1', '2', '3', '4']
    
    bridge_edges = [('124', '201'),
                ('121', '302'),
                ('223', '404'),
                ('324', '401'),
                ('305', '220')]

    node_dict = create_node_dict(node_list)
             
    #joining points within islands
    for letter in island_prefixes:
        for letter_suffix  in flower_graph:
            if letter_suffix < 10:
                first_point_name = letter +'{}{}'.format(0, str(letter_suffix))
            else:
                first_point_name = letter +'{}'.format(letter_suffix)
            first_point_loc = node_dict[first_point_name]
            
            for i in range(0, len(flower_graph[letter_suffix])):
                if flower_graph[letter_suffix][i] < 10:
                    second_point_name = letter + '{}{}'.format(0, flower_graph[letter_suffix][i])
                else:
                    second_point_name = letter + '{}'.format(flower_graph[letter_suffix][i])
                second_point_loc = node_dict[second_point_name]
                
                cv2.line(black_im, first_point_loc, second_point_loc, (255,255,255), 15)
           
            
    #bridging islands             
    for bridges in bridge_edges:
        first_point_name = bridges[0]
        sec_point_name = bridges[1]
        first_point_loc = node_dict[first_point_name]
        sec_point_loc = node_dict[sec_point_name]
      
        cv2.line(black_im , first_point_loc , sec_point_loc , (255,255,255) , 15)
       
        
    #binarizing the mask
    #binarizing the mask
    gray = cv2.cvtColor(black_im, cv2.COLOR_BGR2GRAY)
    arr = np.array(gray)
    mask = arr > 200
    #cv2.imshow('black',gray)
    return mask
    
    #mask = np.zeros((gray.shape),np.uint8)
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    #close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    #div = np.float32(gray)/(close)
    #res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    #res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
    #cv2.imshow('res',gray)
    #cv2.waitKey(0)
    #cv2.imshow('gray',gray)
    #cv2.waitKey(0)
    
    #return res2

from pathlib import Path 
node_list = Path('src/tools/node_list_new.csv').resolve()
create_mask(node_list)
