# -*- coding: utf-8 -*-
"""
Created on Mon May 21 02:23:06 2018
@author: 0x

Create csv file in tensorflow format from xml VOC format
"""


import glob
import pandas as pd
import xml.etree.ElementTree as ET


"""
Create csv file from xml to tensorflow format
"""

# Change path and file name train/test

image_path = "VOC/VOC2012test/Annotations/"
file_name = "test.csv"

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # print (xml_file)
        for member in root.findall('object'):
            if member[0].text == 'person':
                box= member.find('bndbox')
                value = (root.find('filename').text,
                         int(root.find('size').find('width').text),
                         int(root.find('size').find('height').text),
                         member[0].text,
                         int(float(box.find('xmin').text)),
                         int(float(box.find('ymin').text)),
                         int(float(box.find('xmax').text)),
                         int(float(box.find('ymax').text))
                         )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df





def main():
    output_path = "VOC/onlyPersons/"
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(output_path+file_name, index=None)
    print('Successfully converted xml to csv.')



main()