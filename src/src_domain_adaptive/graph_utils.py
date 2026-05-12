import networkx as nx
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
from shapely import geometry
from src.adjacency import AdjacencyEngine
from src.graph_utils import *
from skimage import data, segmentation
from skimage import graph
import matplotlib.pyplot as plt
import networkx as nx
from skimage.segmentation import clear_border
from skimage.segmentation import expand_labels
import json
import cv2
import io
import imageio
import numpy as np
import copy


def draw_rag_on_cells(pred, img, dfx,verbose=0, save_data=False):
    img1=img.copy()
    img2=img.copy()
    msks=pred[0]['masks'].detach().cpu().numpy()
    scrs=pred[0]['scores'].detach().cpu().numpy()
    cell_type=pred[0]['labels'].detach().cpu().numpy()
    boxes=pred[0]['boxes'].detach().cpu().numpy()
    
    pred_label = np.empty((msks[0][0].shape)) 
    cell_types = []
    coords = []
    
    for i in range(0,len(msks)):
        box=boxes[i]
        x = (box[0]+box[2])/2
        y = (box[1]+box[3])/2
        X = int(x)
        Y = int(y)    
        if scrs[i]>0.6 :  #0.9
            pred_label[:,:][msks[i][0]> 0.67] += i+1   #################
            cell_types.append(cell_type[i])
            coords.append((X,Y))
    
    fig, ax = plt.subplots(2,2,figsize=(8,8)) 
    
    pred_label = pred_label.astype('uint8')
    boundary = pred_label
    ax[0,0].imshow(boundary)
    ax[0,0].axis('off')
    
    
    expanded = expand_labels(pred_label, distance=25)
    ax[0,1].imshow(expanded)
    ax[0,1].axis('off')
    
    areas = clear_border(expanded) 
    ax[1,0].imshow(areas)
    ax[1,0].axis('off')
    cell_net = len(np.unique(areas))
    if verbose == 1:
        print(str(cell_net), ' cells you may have in the selected area')
    else:
        pass
    #print(np.unique(areas))   
    g =  graph.rag_mean_color(img1, areas)
    g.remove_node(0)
    lc = graph.show_rag(areas, g, img2, ax=ax[1,1])
    ax[1,1].axis('off')
    #cbar = plt.colorbar(lc)
    plt.tight_layout()
    plt.show()
    
    draw_graph(g, dfx, areas, verbose=verbose)
    


def draw_graph(G, dfx, areas, verbose=1):
    # redefine colors for schematic drawing
    cell_color = [(0,0,0),(1,159/255,155/255),(144/255,229/255,161/255),
                  (161/255,201/255,244/255),(1,180/255,130/255)]
    # add possible attributes for nodes using predicted Panda dataset (dfx)
    for (i,m,n,o) in zip(np.unique(areas),list(dfx['labels']), list(dfx['area']), list(dfx['cell_name'])):
        #if i < cell_net:
        G.add_node(i, color = cell_color[m], cell_type = o, area = n/20, label = m)
        #else:
        #   break # data with low accuracy might be cut off during processing

    for p,q  in list(G.edges): 
        if G.nodes[p]['label'] == 1: #vessel
            x = 1
        elif G.nodes[p]['label'] == 2: #ray
            x = 2
        elif G.nodes[p]['label'] == 3: #fiber
            x = 3
        else: #parencyma
            x = 2
        if G.nodes[q]['label'] == 1:
            y = 1
        elif G.nodes[q]['label'] == 2:
            y = 2
        elif G.nodes[q]['label'] == 3:
            y = 3
        else:
            y = 2
        dist = x * y
        # print(dist, end='.')
        if dist == 3 or dist == 6:
            G.remove_edge(p,q)
            #print('edge ',p,q,' are removed')
        else:
            G.add_edge(p, q, weight=dist)
    
    colors = [G.nodes[node]['color'] for node in list(G.nodes())]
    sizes = [G.nodes[node]['area'] for node in list(G.nodes())]
    edge_width = [ d['weight']*0.4 for (u,v,d) in G.edges(data=True)]
    pos = nx.spring_layout(G)
    #pos = nx.circular_layout(G)
    
    if verbose == 1:
        nx.draw(G, pos, font_size=6, with_labels=True, node_color=colors, node_size=sizes,width=edge_width)
    else:
        nx.draw(G, pos, node_color=colors, node_size=sizes,width=edge_width)

    ####################

    return G




def draw_RAG_Graph_geo(pred, img0, dfx, verbose=0, save_data=False):
    msks=pred[0]['masks'].detach().cpu().numpy()
    scrs=pred[0]['scores'].detach().cpu().numpy()
    cell_type=pred[0]['labels'].detach().cpu().numpy()
    boxes=pred[0]['boxes'].detach().cpu().numpy()
    
    pred_label = np.empty((msks[0][0].shape)) 
    cell_types = []
    coords = []
    
    for i in range(0,len(msks)):
        box=boxes[i]
        x = (box[0]+box[2])/2
        y = (box[1]+box[3])/2
        X = int(x)
        Y = int(y)    
        if scrs[i]>0.6 :  #0.9
            pred_label[:,:][msks[i][0]> 0.67] += i+1   #################
            cell_types.append(cell_type[i])
            coords.append((X,Y))
    
    fig, ax = plt.subplots(2,2,figsize=(8,8)) 
    
    pred_label = pred_label.astype('int8')
    boundary = pred_label
    ax[0,0].imshow(boundary)
    ax[0,0].axis('off')    
    
    expanded = expand_labels(pred_label, distance=25)
    ax[0,1].imshow(expanded)
    ax[0,1].axis('off')
    
    areas = clear_border(expanded) 
    cell_net = len(np.unique(expanded))
    if verbose == 1:
        print(str(cell_net), ' cells you may have in the selected area')
    else:
        pass
    # node diameter is based on cell area
    # edge width is based on cell type
    
    img = np.array(img0)
    ax[1,0].imshow(img0)
    ax[1,0].axis('off')
    g =  graph.rag_mean_color(img, areas)
    g.remove_node(0)
    
    
    lc = graph.show_rag(areas, g, img, ax=ax[1,1])
    ax[1,1].axis('off')
    #cbar = plt.colorbar(lc)
    
    plt.tight_layout()
    plt.show()
    
    if save_data == True:
        buf = io.BytesIO() # prepare buffer
        plt.savefig(buf, format='png') # store in the buffer
        enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # read from buffer
        dst = cv2.imdecode(enc, 1) # decoding
        dst = dst[:,:,::-1] # BGR->RGB
        plt.close()
        return dst
    else:
        plt.show()    
        return G2


def draw_graph_geo(G, dfx, cell_net, verbose=0):
    # redefine colors for schematic drawing
    cell_color = [(0,0,0),(1,159/255,155/255),(144/255,229/255,161/255),\
                  (161/255,201/255,244/255),(1,180/255,130/255)]
    # add possible attributes for nodes using predicted Panda dataset (dfx)
    for i,(m,n,o) in enumerate(zip(list(dfx['labels']), list(dfx['area']), list(dfx['cell_name']))):
        if i < cell_net:
            G.add_node(i, color = cell_color[m], cell_type = o, area = n/20, label = m)
        else:
            break # data with low accuracy might be cut off during processing

    for p,q  in list(G.edges): 
        if G.nodes[p]['label'] == 1: #vessel
            x = 1
        elif G.nodes[p]['label'] == 2: #ray
            x = 2
        elif G.nodes[p]['label'] == 3: #fiber
            x = 3
        else: #parencyma
            x = 2
        if G.nodes[q]['label'] == 1:
            y = 1
        elif G.nodes[q]['label'] == 2:
            y = 2
        elif G.nodes[q]['label'] == 3:
            y = 3
        else:
            y = 2
        dist = x * y
        # print(dist, end='.')
        if dist == 3 or dist == 6:
            G.remove_edge(p,q)
            #print('edge ',p,q,' are removed')
        else:
            G.add_edge(p, q, weight=dist)
    
    colors = [G.nodes[node]['color'] for node in list(G.nodes())]
    sizes = [G.nodes[node]['area'] for node in list(G.nodes())]
    edge_width = [ d['weight']*0.4 for (u,v,d) in G.edges(data=True)]
    pos = nx.spring_layout(G)
    #pos = nx.circular_layout(G)
    
    if verbose == 1:
        nx.draw(G, pos, font_size=6, with_labels=True, node_color=colors, node_size=sizes,width=edge_width)
    else:
        nx.draw(G, pos, node_color=colors, node_size=sizes,width=edge_width)

    ####################

    return G
