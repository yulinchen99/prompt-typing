#!/bin/sh
data_dir=`dirname $0`
wget -O $data_dir/samples.zip https://cloud.tsinghua.edu.cn/f/ac665b7ad71e4bd19eb6/?dl=1
unzip -d $data_dir $data_dir/samples.zip