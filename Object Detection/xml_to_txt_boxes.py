# -*- coding: utf-8 -*-
"""
Created on Mon May 21 02:23:06 2018
@author: 0x

Create txt file in YOLO format from xml VOC format
"""


import glob
import pandas as pd
import xml.etree.ElementTree as ET



# Change routes to test directory to create txt for the test files (output, image_path and txt name)



def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # print (xml_file)
        file_name = root.find('filename').text
        output = "VOC/onlyPerson/train/{}".format(file_name)
        for member in root.findall('object'):
            if member[0].text == 'person':
                box= member.find('bndbox')
                xmin = int(float(box.find('xmin').text))
                ymin = int(float(box.find('ymin').text))
                xmax = int(float(box.find('xmax').text))
                ymax = int(float(box.find('ymax').text))
                output += " {},{},{},{},0".format(xmin, ymin, xmax, ymax)
            if len(output)>40:
                xml_list.append(output)

                

    xml_df = pd.DataFrame(xml_list)
    return xml_df





def main():
    image_path = "VOC/VOC2012train/Annotations/"
    output_path = "VOC/onlyPersons/"
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(output_path+'yolo_train.txt', header=None, sep=' ', index=None)
    print('Successfully converted xml to csv.')


########### Remove Double quotes from file #######
main()

