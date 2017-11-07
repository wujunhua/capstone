from flask import Flask, render_template, url_for, redirect
from app import app
import pandas as pd
import numpy as np
import glob
import json


@app.route('/')
def index():
    """ Homepage """
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
                Nose.append(parts['pose_keypoints'][0:3])
                Neck.append(parts['pose_keypoints'][3:6])
                RShoulder.append(parts['pose_keypoints'][6:9])
                RElbow.append(parts['pose_keypoints'][9:12])
                RWrist.append(parts['pose_keypoints'][12:15])
                LShoulder.append(parts['pose_keypoints'][15:18])
                LElbow.append(parts['pose_keypoints'][18:21])
                LWrist.append(parts['pose_keypoints'][21:24])
                RHip.append(parts['pose_keypoints'][24:27])
                RKnee.append(parts['pose_keypoints'][27:30])
                RAnkle.append(parts['pose_keypoints'][30:33])
                LHip.append(parts['pose_keypoints'][33:36])
                LKnee.append(parts['pose_keypoints'][36:39])
                LAnkle.append(parts['pose_keypoints'][39:42])
                REye.append(parts['pose_keypoints'][42:45])
                LEye.append(parts['pose_keypoints'][45:48])
                REar.append(parts['pose_keypoints'][48:51])
                LEar.append(parts['pose_keypoints'][51:54])

    bConfidence = pd.DataFrame({'nose': Nose,
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



    partNose        = np.array(Nose)
    partNeck        = np.array(Neck)
    partRShoulder   = np.array(RShoulder)
    partRElbow      = np.array(RElbow)
    partRWrist      = np.array(RWrist)
    partLShoulder   = np.array(LShoulder)
    partLElbow      = np.array(LElbow)
    partLWrist      = np.array(LWrist)
    partRHip        = np.array(RHip)
    partRKnee       = np.array(RKnee)
    partRAnkle      = np.array(RAnkle)
    partLHip        = np.array(LHip)
    partLKnee       = np.array(LKnee)
    partLAnkle      = np.array(LAnkle)
    partREye        = np.array(REye)
    partLEye        = np.array(LEye)
    partREar        = np.array(REar)
    partLEar        = np.array(LEar)

    avgNose        = sum(partNose[:,2])/len(partNose[:,2])
    avgNeck        = sum(partNeck[:,2])/len(partNeck[:,2])
    avgRShoulder   = sum(partRShoulder[:,2])/len(partRShoulder[:,2])
    avgRElbow      = sum(partRElbow[:,2])/len(partRElbow[:,2])
    avgRWrist      = sum(partRWrist[:,2])/len(partRWrist[:,2])
    avgLShoulder   = sum(partLShoulder[:,2])/len(partLShoulder[:,2])
    avgLElbow      = sum(partLElbow[:,2])/len(partLElbow[:,2])
    avgLWrist      = sum(partLWrist[:,2])/len(partLWrist[:,2])
    avgRHip        = sum(partRHip[:,2])/len(partRHip[:,2])
    avgRKnee       = sum(partRKnee[:,2])/len(partRKnee[:,2])
    avgRAnkle      = sum(partRAnkle[:,2])/len(partRAnkle[:,2])
    avgLHip        = sum(partLHip[:,2])/len(partLHip[:,2])
    avgLKnee       = sum(partLKnee[:,2])/len(partLKnee[:,2])
    avgLAnkle      = sum(partLAnkle[:,2])/len(partLAnkle[:,2])
    avgREye        = sum(partREye[:,2])/len(partREye[:,2])
    avgLEye        = sum(partLEye[:,2])/len(partLEye[:,2])
    avgREar        = sum(partREar[:,2])/len(partREar[:,2])
    avgLEar        = sum(partLEar[:,2])/len(partLEar[:,2])


    bavgConf = pd.DataFrame({'nose': avgNose,
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

    bConfidence = bConfidence.to_html()
    bavgConf = bavgConf.to_html()

    return render_template('baseball.html', bConfidence = bConfidence, bavgConf = bavgConf)

@app.route('/cycling.html')
def cycling():
    folder = "app/static/json/sport/cycling/cycling1/*.json"
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
                Nose.append(parts['pose_keypoints'][0:3])
                Neck.append(parts['pose_keypoints'][3:6])
                RShoulder.append(parts['pose_keypoints'][6:9])
                RElbow.append(parts['pose_keypoints'][9:12])
                RWrist.append(parts['pose_keypoints'][12:15])
                LShoulder.append(parts['pose_keypoints'][15:18])
                LElbow.append(parts['pose_keypoints'][18:21])
                LWrist.append(parts['pose_keypoints'][21:24])
                RHip.append(parts['pose_keypoints'][24:27])
                RKnee.append(parts['pose_keypoints'][27:30])
                RAnkle.append(parts['pose_keypoints'][30:33])
                LHip.append(parts['pose_keypoints'][33:36])
                LKnee.append(parts['pose_keypoints'][36:39])
                LAnkle.append(parts['pose_keypoints'][39:42])
                REye.append(parts['pose_keypoints'][42:45])
                LEye.append(parts['pose_keypoints'][45:48])
                REar.append(parts['pose_keypoints'][48:51])
                LEar.append(parts['pose_keypoints'][51:54])

    cConfidence = pd.DataFrame({'nose': Nose,
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



    partNose        = np.array(Nose)
    partNeck        = np.array(Neck)
    partRShoulder   = np.array(RShoulder)
    partRElbow      = np.array(RElbow)
    partRWrist      = np.array(RWrist)
    partLShoulder   = np.array(LShoulder)
    partLElbow      = np.array(LElbow)
    partLWrist      = np.array(LWrist)
    partRHip        = np.array(RHip)
    partRKnee       = np.array(RKnee)
    partRAnkle      = np.array(RAnkle)
    partLHip        = np.array(LHip)
    partLKnee       = np.array(LKnee)
    partLAnkle      = np.array(LAnkle)
    partREye        = np.array(REye)
    partLEye        = np.array(LEye)
    partREar        = np.array(REar)
    partLEar        = np.array(LEar)

    avgNose        = sum(partNose[:,2])/len(partNose[:,2])
    avgNeck        = sum(partNeck[:,2])/len(partNeck[:,2])
    avgRShoulder   = sum(partRShoulder[:,2])/len(partRShoulder[:,2])
    avgRElbow      = sum(partRElbow[:,2])/len(partRElbow[:,2])
    avgRWrist      = sum(partRWrist[:,2])/len(partRWrist[:,2])
    avgLShoulder   = sum(partLShoulder[:,2])/len(partLShoulder[:,2])
    avgLElbow      = sum(partLElbow[:,2])/len(partLElbow[:,2])
    avgLWrist      = sum(partLWrist[:,2])/len(partLWrist[:,2])
    avgRHip        = sum(partRHip[:,2])/len(partRHip[:,2])
    avgRKnee       = sum(partRKnee[:,2])/len(partRKnee[:,2])
    avgRAnkle      = sum(partRAnkle[:,2])/len(partRAnkle[:,2])
    avgLHip        = sum(partLHip[:,2])/len(partLHip[:,2])
    avgLKnee       = sum(partLKnee[:,2])/len(partLKnee[:,2])
    avgLAnkle      = sum(partLAnkle[:,2])/len(partLAnkle[:,2])
    avgREye        = sum(partREye[:,2])/len(partREye[:,2])
    avgLEye        = sum(partLEye[:,2])/len(partLEye[:,2])
    avgREar        = sum(partREar[:,2])/len(partREar[:,2])
    avgLEar        = sum(partLEar[:,2])/len(partLEar[:,2])


    cavgConf = pd.DataFrame({'nose': avgNose,
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

    cConfidence = cConfidence.to_html()
    cavgConf = cavgConf.to_html()

    return render_template('cycling.html', cConfidence = cConfidence, cavgConf = cavgConf)

@app.route('/compare.html')
def compare():
    folder1 = "app/static/json/sport/baseball/baseball1/*.json"
    folder2 = "app/static/json/sport/cycling/cycling1/*.json"

    files1 = glob.glob(folder1)
    files2 = glob.glob(folder2)

    bNose        = []
    bNeck        = []
    bRShoulder   = []
    bRElbow      = []
    bRWrist      = []
    bLShoulder   = []
    bLElbow      = []
    bLWrist      = []
    bRHip        = []
    bRKnee       = []
    bRAnkle      = []
    bLHip        = []
    bLKnee       = []
    bLAnkle      = []
    bREye        = []
    bLEye        = []
    bREar        = []
    bLEar        = []

    cNose        = []
    cNeck        = []
    cRShoulder   = []
    cRElbow      = []
    cRWrist      = []
    cLShoulder   = []
    cLElbow      = []
    cLWrist      = []
    cRHip        = []
    cRKnee       = []
    cRAnkle      = []
    cLHip        = []
    cLKnee       = []
    cLAnkle      = []
    cREye        = []
    cLEye        = []
    cREar        = []
    cLEar        = []

    for file in files1:
        with open(file) as json_data:
            data = json.load(json_data)
            for parts in data['people']:
                bNose.append(parts['pose_keypoints'][2])
                bNeck.append(parts['pose_keypoints'][5])
                bRShoulder.append(parts['pose_keypoints'][8])
                bRElbow.append(parts['pose_keypoints'][11])
                bRWrist.append(parts['pose_keypoints'][14])
                bLShoulder.append(parts['pose_keypoints'][17])
                bLElbow.append(parts['pose_keypoints'][20])
                bLWrist.append(parts['pose_keypoints'][23])
                bRHip.append(parts['pose_keypoints'][26])
                bRKnee.append(parts['pose_keypoints'][29])
                bRAnkle.append(parts['pose_keypoints'][32])
                bLHip.append(parts['pose_keypoints'][35])
                bLKnee.append(parts['pose_keypoints'][38])
                bLAnkle.append(parts['pose_keypoints'][41])
                bREye.append(parts['pose_keypoints'][44])
                bLEye.append(parts['pose_keypoints'][47])
                bREar.append(parts['pose_keypoints'][50])
                bLEar.append(parts['pose_keypoints'][53])


    for file in files2:
        with open(file) as json_data:
            data = json.load(json_data)
            for parts in data['people']:
                cNose.append(parts['pose_keypoints'][2])
                cNeck.append(parts['pose_keypoints'][5])
                cRShoulder.append(parts['pose_keypoints'][8])
                cRElbow.append(parts['pose_keypoints'][11])
                cRWrist.append(parts['pose_keypoints'][14])
                cLShoulder.append(parts['pose_keypoints'][17])
                cLElbow.append(parts['pose_keypoints'][20])
                cLWrist.append(parts['pose_keypoints'][23])
                cRHip.append(parts['pose_keypoints'][26])
                cRKnee.append(parts['pose_keypoints'][29])
                cRAnkle.append(parts['pose_keypoints'][32])
                cLHip.append(parts['pose_keypoints'][35])
                cLKnee.append(parts['pose_keypoints'][38])
                cLAnkle.append(parts['pose_keypoints'][41])
                cREye.append(parts['pose_keypoints'][44])
                cLEye.append(parts['pose_keypoints'][47])
                cREar.append(parts['pose_keypoints'][50])
                cLEar.append(parts['pose_keypoints'][53])

    bavgNose        = sum(bNose)/len(bNose)
    bavgNeck        = sum(bNeck)/len(bNeck)
    bavgRShoulder   = sum(bRShoulder)/len(bRShoulder)
    bavgRElbow      = sum(bRElbow)/len(bRElbow)
    bavgRWrist      = sum(bRWrist)/len(bRWrist)
    bavgLShoulder   = sum(bLShoulder)/len(bLShoulder)
    bavgLElbow      = sum(bLElbow)/len(bLElbow)
    bavgLWrist      = sum(bLWrist)/len(bLWrist)
    bavgRHip        = sum(bRHip)/len(bRHip)
    bavgRKnee       = sum(bRKnee)/len(bRKnee)
    bavgRAnkle      = sum(bRAnkle)/len(bRAnkle)
    bavgLHip        = sum(bLHip)/len(bLHip)
    bavgLKnee       = sum(bLKnee)/len(bLKnee)
    bavgLAnkle      = sum(bLAnkle)/len(bLAnkle)
    bavgREye        = sum(bREye)/len(bREye)
    bavgLEye        = sum(bLEye)/len(bLEye)
    bavgREar        = sum(bREar)/len(bREar)
    bavgLEar        = sum(bLEar)/len(bLEar)

    cavgNose        = sum(cNose)/len(cNose)
    cavgNeck        = sum(cNeck)/len(cNeck)
    cavgRShoulder   = sum(cRShoulder)/len(cRShoulder)
    cavgRElbow      = sum(cRElbow)/len(cRElbow)
    cavgRWrist      = sum(cRWrist)/len(cRWrist)
    cavgLShoulder   = sum(cLShoulder)/len(cLShoulder)
    cavgLElbow      = sum(cLElbow)/len(cLElbow)
    cavgLWrist      = sum(cLWrist)/len(cLWrist)
    cavgRHip        = sum(cRHip)/len(cRHip)
    cavgRKnee       = sum(cRKnee)/len(cRKnee)
    cavgRAnkle      = sum(cRAnkle)/len(cRAnkle)
    cavgLHip        = sum(cLHip)/len(cLHip)
    cavgLKnee       = sum(cLKnee)/len(cLKnee)
    cavgLAnkle      = sum(cLAnkle)/len(cLAnkle)
    cavgREye        = sum(cREye)/len(cREye)
    cavgLEye        = sum(cLEye)/len(cLEye)
    cavgREar        = sum(cREar)/len(cREar)
    cavgLEar        = sum(cLEar)/len(cLEar)


    bavgConf = pd.DataFrame({'nose': bavgNose,
                            'neck': bavgNeck,
                            'rshoulder': bavgRShoulder,
                            'relbow': bavgRElbow,
                            'rwrist': bavgRWrist,
                            'lshoulder': bavgLShoulder,
                            'lelbow': bavgLElbow,
                            'lwrist': bavgLWrist,
                            'rhip': bavgRHip,
                            'rknee': bavgRKnee,
                            'rankle': bavgRAnkle,
                            'lhip': bavgLHip,
                            'lknee': bavgLKnee,
                            'lankle': bavgLAnkle,
                            'reye': bavgREye,
                            'leye': bavgLEye,
                            'rear': bavgREar,
                            'lear': bavgLEar},
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


    cavgConf = pd.DataFrame({'nose': cavgNose,
                            'neck': cavgNeck,
                            'rshoulder': cavgRShoulder,
                            'relbow': cavgRElbow,
                            'rwrist': cavgRWrist,
                            'lshoulder': cavgLShoulder,
                            'lelbow': cavgLElbow,
                            'lwrist': cavgLWrist,
                            'rhip': cavgRHip,
                            'rknee': cavgRKnee,
                            'rankle': cavgRAnkle,
                            'lhip': cavgLHip,
                            'lknee': cavgLKnee,
                            'lankle': cavgLAnkle,
                            'reye': cavgREye,
                            'leye': cavgLEye,
                            'rear': cavgREar,
                            'lear': cavgLEar},
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

    bavgConf = bavgConf.to_html()
    cavgConf = cavgConf.to_html()
    return render_template('compare.html', bavgConf = bavgConf, cavgConf = cavgConf)
