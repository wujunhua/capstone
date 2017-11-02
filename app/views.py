from flask import Flask, render_template
from app import app
import pandas as pd
import numpy as np
import glob
import json


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/baseball.html')
def baseball():
    folder = "app/static/json/sport/baseball/baseball1/*.json"
    files = glob.glob(folder)

    Nose        = []
    Neck        = []
    RShoulder   = []
    RElbow      = []
    RWrist      = []
    LShoulder   = []
    LElbow      = []
    LWrist      = []
    RHip        = []
    RKnee       = []
    RAnkle      = []
    LHip        = []
    LKnee       = []
    LAnkle      = []
    REye        = []
    LEye        = []
    REar        = []
    LEar        = []

    for file in files:
        with open(file) as json_data:
            data = json.load(json_data)
            for parts in data['people']:
                Nose.append(parts['pose_keypoints'][2])
                Neck.append(parts['pose_keypoints'][5])
                RShoulder.append(parts['pose_keypoints'][8])
                RElbow.append(parts['pose_keypoints'][11])
                RWrist.append(parts['pose_keypoints'][14])
                LShoulder.append(parts['pose_keypoints'][17])
                LElbow.append(parts['pose_keypoints'][20])
                LWrist.append(parts['pose_keypoints'][23])
                RHip.append(parts['pose_keypoints'][26])
                RKnee.append(parts['pose_keypoints'][29])
                RAnkle.append(parts['pose_keypoints'][32])
                LHip.append(parts['pose_keypoints'][35])
                LKnee.append(parts['pose_keypoints'][38])
                LAnkle.append(parts['pose_keypoints'][41])
                REye.append(parts['pose_keypoints'][44])
                LEye.append(parts['pose_keypoints'][47])
                REar.append(parts['pose_keypoints'][50])
                LEar.append(parts['pose_keypoints'][53])

    '''
    print(pd.DataFrame({'nose': Nose,
                        'neck': Neck,
                        'rshoulder': RShoulder,
                        'relbow': RElbow,
                        'rwrist': RWrist,
                        'lshoulder': LShoulder,
                        'lelbow': LElbow,
                        'lwrist': LWrist,
                        'rhip': RHip,
                        'rknee': RKnee,
                        'rankle': RAnkle,
                        'lhip': LHip,
                        'lknee': LKnee,
                        'lankle': LAnkle,
                        'reye': REye,
                        'leye': LEye,
                        'rear': REar,
                        'lear': LEar}))
    '''

    Confidence = pd.DataFrame({'nose': Nose,
                               'neck': Neck,
                               'rshoulder': RShoulder,
                               'relbow': RElbow,
                               'rwrist': RWrist,
                               'lshoulder': LShoulder,
                               'lelbow': LElbow,
                               'lwrist': LWrist,
                               'rhip': RHip,
                               'rknee': RKnee,
                               'rankle': RAnkle,
                               'lhip': LHip,
                               'lknee': LKnee,
                               'lankle': LAnkle,
                               'reye': REye,
                               'leye': LEye,
                               'rear': REar,
                               'lear': LEar},
                               columns=['nose',
                                       'neck',
                                       'rshoulder',
                                       'relbow',
                                       'rwrist',
                                       'lshoulder',
                                       'lelbow',
                                       'lwrist',
                                       'rhip',
                                       'rknee',
                                       'rankle',
                                       'lhip',
                                       'lknee',
                                       'lankle',
                                       'reye',
                                       'leye',
                                       'rear',
                                       'lear'])

    avgNose        = sum(Nose)/len(Nose)
    avgNeck        = sum(Neck)/len(Neck)
    avgRShoulder   = sum(RShoulder)/len(RShoulder)
    avgRElbow      = sum(RElbow)/len(RElbow)
    avgRWrist      = sum(RWrist)/len(RWrist)
    avgLShoulder   = sum(LShoulder)/len(LShoulder)
    avgLElbow      = sum(LElbow)/len(LElbow)
    avgLWrist      = sum(LWrist)/len(LWrist)
    avgRHip        = sum(RHip)/len(RHip)
    avgRKnee       = sum(RKnee)/len(RKnee)
    avgRAnkle      = sum(RAnkle)/len(RAnkle)
    avgLHip        = sum(LHip)/len(LHip)
    avgLKnee       = sum(LKnee)/len(LKnee)
    avgLAnkle      = sum(LAnkle)/len(LAnkle)
    avgREye        = sum(REye)/len(REye)
    avgLEye        = sum(LEye)/len(LEye)
    avgREar        = sum(REar)/len(REar)
    avgLEar        = sum(LEar)/len(LEar)


    '''
    print(pd.DataFrame({'nose': avgNose,
                        'neck': avgNeck,
                        'rshoulder': avgRShoulder,
                        'relbow': avgRElbow,
                        'rwrist': avgRWrist,
                        'lshoulder': avgLShoulder,
                        'lelbow': avgLElbow,
                        'lwrist': avgLWrist,
                        'rhip': avgRHip,
                        'rknee': avgRKnee,
                        'rankle': avgRAnkle,
                        'lhip': avgLHip,
                        'lknee': avgLKnee,
                        'lankle': avgLAnkle,
                        'reye': avgREye,
                        'leye': avgLEye,
                        'rear': avgREar,
                        'lear': avgLEar},
                        index=['Confidence Average'],
                        columns=['nose',
                                 'neck',
                                 'rshoulder',
                                 'relbow',
                                 'rwrist',
                                 'lshoulder',
                                 'lelbow',
                                 'lwrist',
                                 'rhip',
                                 'rknee',
                                 'rankle',
                                 'lhip',
                                 'lknee',
                                 'lankle',
                                 'reye',
                                 'leye',
                                 'rear',
                                 'lear']))
    '''

    avgConf = pd.DataFrame({'nose': avgNose,
                            'neck': avgNeck,
                            'rshoulder': avgRShoulder,
                            'relbow': avgRElbow,
                            'rwrist': avgRWrist,
                            'lshoulder': avgLShoulder,
                            'lelbow': avgLElbow,
                            'lwrist': avgLWrist,
                            'rhip': avgRHip,
                            'rknee': avgRKnee,
                            'rankle': avgRAnkle,
                            'lhip': avgLHip,
                            'lknee': avgLKnee,
                            'lankle': avgLAnkle,
                            'reye': avgREye,
                            'leye': avgLEye,
                            'rear': avgREar,
                            'lear': avgLEar},
                            index=['Avg Conf'],
                            columns=['nose',
                                    'neck',
                                    'rshoulder',
                                    'relbow',
                                    'rwrist',
                                    'lshoulder',
                                    'lelbow',
                                    'lwrist',
                                    'rhip',
                                    'rknee',
                                    'rankle',
                                    'lhip',
                                    'lknee',
                                    'lankle',
                                    'reye',
                                    'leye',
                                    'rear',
                                    'lear'])

    return render_template('baseball.html', Confidence = Confidence.to_html(), avgConf = avgConf.to_html())
