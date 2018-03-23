import pandas as pd
import os
import tarfile
from six.moves import urllib
#Date-of-code: 2018-03-23
#Author: datnt
#Descriptions:
#Give a url to a file stored in web
#Download and store in disk
#=====================================================================================#
DOWNLOAD_ROOT = "https://github.com/idahousa/sampleCodes/tree/master/datasets/housing.tgz"
#=======Load data to work space========================#
def fetch_housing_data(housing_url = DOWNLOAD_ROOT, save_dir = "datasets"):
    current_dir_path = os.getcwd()
    save_dir_path = os.path.join(current_dir_path,save_dir)
    if not os.path.isdir(save_dir_path):
        os.makedirs(save_dir_path)
    save_file_path = save_dir_path + '/housing.tgz'
    #Download file and save to disk
    urllib.request.urlretrieve(housing_url,save_file_path)
    #Extract the downloaded file
    housing_tgz = tarfile.open(save_file_path)
    housing_tgz.extractall(save_dir_path)
    housing_tgz.close()
    return save_file_path
#=====================================================================================#
