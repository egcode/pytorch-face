from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import requests
import zipfile
import os
import sys
from pdb import set_trace as bp
from tqdm import tqdm
import math

data_dict = {
    'lfw_160':'1KxMtqYMhYy2jU6q9DZqoc0_cFL6Md0gE', 
    'CASIA_Webface_160':'175YhXe26wMMxSRuKGAbbVCkY5MLDk5m7', 
    'go' : '1EFdYOLvQY63-bBPoKJz79XUl-QiZll4c',
    'golovan_160' : '1AUVdEfRy1lelj9xYhkTudUP47SsfGFVU',
    'CASIA_and_Golovan_160' : '1Z1nzXX9KxUUjGauc5hHcj9dwgybsecNj'
    }

def download_and_extract_file(model_name, data_dir):
    file_id = data_dict[model_name]
    destination = os.path.join(data_dir, model_name + '.zip')
    if not os.path.exists(destination):
        print('Downloading file to %s' % destination)
        download_file_from_google_drive(file_id, destination)
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            print('Extracting file to %s' % data_dir)
            zip_ref.extractall(data_dir)

def download_file_from_google_drive(file_id, destination):
    
        URL = "https://drive.google.com/uc?export=download"

        session = requests.Session()
    
        print("Downloading %s" % destination)
        headers = {'Range':'bytes=0-'}
        r = session.get(URL,headers=headers, params = { 'id' : file_id }, stream = True)

        token = get_confirm_token(r)
        if token:
            params = { 'id' : file_id, 'confirm' : token }
            r = session.get(URL,headers=headers, params = params, stream = True)
            save_response_content(r, destination)
        else:
            save_response_content(r, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(r, destination):
    rr = r.headers['Content-Range']
    total_length=int(rr.partition('/')[-1])

    block_size = 32768
    wrote = 0 
    with open(destination, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_length//block_size) , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_length != 0 and wrote != total_length:
        print("ERROR, something went wrong")  


if __name__ == '__main__':
    
    out_dir = 'data/'
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)

    # download_and_extract_file('go', out_dir)
    download_and_extract_file('lfw_160', out_dir)
    # download_and_extract_file('CASIA_Webface_160', out_dir)
    download_and_extract_file('CASIA_and_Golovan_160', out_dir)
