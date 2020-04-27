# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:54:09 2020

@author: chinging
"""

# -*- coding: utf-8 -*-
 
import wget, tarfile
import os
 
DATA_URL = 'https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=1'
out_fname = 'abc.tar.gz'

 
 
wget.download(DATA_URL, out=out_fname)
# 提取压缩包
tar = tarfile.open(out_fname)
tar.extractall()
tar.close()
# 删除下载文件
os.remove(out_fname)
