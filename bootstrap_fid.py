
# 1. USER DEFFINE ORIG FOLDER AND TESTED ONE
# 2. USER DEFINE K - NUMBER OF IMAGES TO BE SAMPLE
# 3. USER DEFINE N - NUMBER OF RUNS
# 4. CREATE TO TEMP FOLDER AND COPY K FILES TO THEM
# 5. CALCULATE FID FOR EACH
# 6. REPEAT N TIMES
# 7. CALC MEAN AND VAR ON THIS VECTOR


import os
import argparse 
import random 
import shutil
import numpy as np
import time 

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--num_on_images_in_subset', default=1000, required=True, type=int)
parser.add_argument('-n', '--num_of_iter', default=5, required=True, type=int)
parser.add_argument('-p1', '--path_source', required=True, type=str)
parser.add_argument('-p2', '--path_test', required=True, type=str)
parser.add_argument('-g', '--gpu_num', required=True,default="0", type=str)
args = parser.parse_args()

k = args.num_on_images_in_subset
n = args.num_of_iter
path_source_orig = args.path_source
path_test_orig = args.path_test
gpu_num = args.gpu_num

curr_iter_name = path_test_orig.split('/')[-2]
cwd = os.getcwd()
source_str = curr_iter_name + "_tmp_source"
test_str = curr_iter_name + "_tmp_test"
path_source_dst = os.path.join(cwd, source_str)
path_test_dst = os.path.join(cwd, test_str)
fid_scores_arr = []

if os.path.exists(path_source_dst):
  shutil.rmtree(path_source_dst)
  
if os.path.exists(path_test_dst):
  shutil.rmtree(path_test_dst)

for j in range(n):

  print("Start interation " + str(j) + " out of " + str(n))
  
  os.mkdir(path_test_dst) # For test dataset
  os.mkdir(path_source_dst) # For source dataset
  
  #Using for loop to randomly choose multiple files
  print("Extract " + str(k) + " Files to tmp folder")
  t = t = time.time()
  
  for i in range(k):
      
      #Variable random_file stores the name of the random file chosen
      
      random_file_source=random.choice(os.listdir(path_source_orig))
      random_file_test=random.choice(os.listdir(path_test_orig))
      source_file=os.path.join(path_source_orig,random_file_source)
      test_file=os.path.join(path_test_orig,random_file_test)
      
      #"shutil.copy" function copy file from one directory to another
      
      random_file_source = random_file_source.split('.')[0]
      random_file_test = random_file_test.split('.')[0]
      path_test_file = path_test_dst + "/" + random_file_test + "_" + str(j) + ".png"
      path_source_file = path_source_dst + "/" + random_file_source + "_" + str(j) + ".png"
      shutil.copy(source_file,path_source_file)
      shutil.copy(test_file,path_test_file)
      
  
  
  elapsed = time.time() - t
  print("time elapsed (move folders): " + str(elapsed))
  t = t = time.time()
  
  # Calculate FID Score for current iteration : 
  
  stream = os.popen("CUDA_VISIBLE_DEVICES=" + gpu_num + " python -m pytorch_fid " + path_test_dst + " " + path_source_orig + " --device cuda:0")
  output = stream.read()
  output_arr_split = output.split()
  
  i=0
  
  # Parse score to float variable and append to lis : 
  
  for str_tmp in output_arr_split:
    if str_tmp == "FID:":
      i+=1
      fid_curr_val = float(output_arr_split[i])
      print("FID curr val : " + str(fid_curr_val))
      break
    i+=1
  fid_scores_arr.append(fid_curr_val)
  
  # Delete temporary folder : 
  
  shutil.rmtree(path_test_dst)
  shutil.rmtree(path_source_dst)
  elapsed = time.time() - t
  print("time elapsed (calculate FID): " + str(elapsed))
  
  
# Calculate statistics over all results : 

np_fid_scores = np.array(fid_scores_arr)
mu = np.mean(np_fid_scores)
var = np.var(np_fid_scores)
res = np.array((mu,var))
fname = curr_iter_name + "_fid_score.txt"
cwd = os.getcwd()
path = os.path.join(cwd,fname)
np.savetxt(path,res)
print(str(res))
print("Finish calc FID bootstrap")