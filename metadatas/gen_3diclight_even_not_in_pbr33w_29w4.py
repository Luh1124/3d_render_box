import os  
import glob  
import hashlib  
import pandas as pd  
import ast  
from concurrent.futures import ProcessPoolExecutor, as_completed  
from concurrent.futures import ThreadPoolExecutor, as_completed  

from tqdm import tqdm  

three_w_path = "/baai-cwm-backup/hong.li/data/sketchfab/sketchfab/hf-objaverse-v1" 

even_dir = "/baai-cwm-backup/cwm/hong.li/data/3diclight_rendering/3diclight_even_not_in_pbr33w_29w4/eevee_all"

