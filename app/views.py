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
    folder = "app/static/json/sport/baseball/*.json"
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

    hashTable = {}
    collision = []
    collisionLocation = []
    collisionCount = 0

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
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in hashTable:
                            hashTable[current] += 1
                            collisionLocation.append(current)
                            collisionCount += 1

                        else:
                            hashTable[current] = 0
            hashTable = {}
            collision.append(collisionCount)
            collisionCount = 0

            if collision != 0:
                collisionLocation.append(current)

    getCount = np.array(collision)
    getLocation = np.array(collisionLocation)

    mashCount = pd.DataFrame(getCount, columns=['# of Occlusion'])
    mashLocation = pd.DataFrame(getLocation, columns=['Location of Occlusion'])

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

    return render_template('baseball.html', bConfidence = bConfidence, bavgConf = bavgConf,  mashCount = mashCount.to_html(), mashLocation = mashLocation.to_html())

@app.route('/basketball.html')
def basketball():
    folder = "app/static/json/sport/basketball/*.json"
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

    hashTable = {}
    collision = []
    collisionLocation = []
    collisionCount = 0

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
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in hashTable:
                            hashTable[current] += 1
                            collisionLocation.append(current)
                            collisionCount += 1

                        else:
                            hashTable[current] = 0
            hashTable = {}
            collision.append(collisionCount)
            collisionCount = 0

            if collision != 0:
                collisionLocation.append(current)

    getCount = np.array(collision)
    getLocation = np.array(collisionLocation)

    mashCount = pd.DataFrame(getCount, columns=['# of Occlusion'])
    mashLocation = pd.DataFrame(getLocation, columns=['Location of Occlusion'])

    bbConfidence = pd.DataFrame({'nose': Nose,
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


    bbavgConf = pd.DataFrame({'nose': avgNose,
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

    bbConfidence = bbConfidence.to_html()
    bbavgConf = bbavgConf.to_html()

    return render_template('basketball.html', bbConfidence = bbConfidence, bbavgConf = bbavgConf, mashCount = mashCount.to_html(), mashLocation = mashLocation.to_html())

@app.route('/cycling.html')
def cycling():
    folder = "app/static/json/sport/cycling/*.json"
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

    hashTable = {}
    collision = []
    collisionLocation = []
    collisionCount = 0

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
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in hashTable:
                            hashTable[current] += 1
                            collisionLocation.append(current)
                            collisionCount += 1

                        else:
                            hashTable[current] = 0
            hashTable = {}
            collision.append(collisionCount)
            collisionCount = 0

            if collision != 0:
                collisionLocation.append(current)

    getCount = np.array(collision)
    getLocation = np.array(collisionLocation)

    mashCount = pd.DataFrame(getCount, columns=['# of Occlusion'])
    mashLocation = pd.DataFrame(getLocation, columns=['Location of Occlusion'])

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

    return render_template('cycling.html', cConfidence = cConfidence, cavgConf = cavgConf, mashCount = mashCount.to_html(), mashLocation = mashLocation.to_html())

@app.route('/running.html')
def running():
    folder = "app/static/json/sport/running/*.json"
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

    hashTable = {}
    collision = []
    collisionLocation = []
    collisionCount = 0

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
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in hashTable:
                            hashTable[current] += 1
                            collisionLocation.append(current)
                            collisionCount += 1

                        else:
                            hashTable[current] = 0
            hashTable = {}
            collision.append(collisionCount)
            collisionCount = 0

            if collision != 0:
                collisionLocation.append(current)

    getCount = np.array(collision)
    getLocation = np.array(collisionLocation)

    mashCount = pd.DataFrame(getCount, columns=['# of Occlusion'])
    mashLocation = pd.DataFrame(getLocation, columns=['Location of Occlusion'])

    rConfidence = pd.DataFrame({'nose': Nose,
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


    ravgConf = pd.DataFrame({'nose': avgNose,
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

    rConfidence = rConfidence.to_html()
    ravgConf = ravgConf.to_html()

    return render_template('running.html', rConfidence = rConfidence, ravgConf = ravgConf, mashCount = mashCount.to_html(), mashLocation = mashLocation.to_html())

@app.route('/tabletennis.html')
def tabletennis():
    folder = "app/static/json/sport/tabletennis/*.json"
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

    hashTable = {}
    collision = []
    collisionLocation = []
    collisionCount = 0

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
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in hashTable:
                            hashTable[current] += 1
                            collisionLocation.append(current)
                            collisionCount += 1

                        else:
                            hashTable[current] = 0
            hashTable = {}
            collision.append(collisionCount)
            collisionCount = 0

            if collision != 0:
                collisionLocation.append(current)

    getCount = np.array(collision)
    getLocation = np.array(collisionLocation)

    mashCount = pd.DataFrame(getCount, columns=['# of Occlusion'])
    mashLocation = pd.DataFrame(getLocation, columns=['Location of Occlusion'])

    tConfidence = pd.DataFrame({'nose': Nose,
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


    tavgConf = pd.DataFrame({'nose': avgNose,
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

    tConfidence = tConfidence.to_html()
    tavgConf = tavgConf.to_html()

    return render_template('tabletennis.html', tConfidence = tConfidence, tavgConf = tavgConf, mashCount = mashCount.to_html(), mashLocation = mashLocation.to_html())

@app.route('/compare.html')
def compare():
    folder1 = "app/static/json/sport/baseball/*.json"
    folder2 = "app/static/json/sport/basketball/*.json"
    folder3 = "app/static/json/sport/cycling/*.json"
    folder4 = "app/static/json/sport/running/*.json"
    folder5 = "app/static/json/sport/tabletennis/*.json"

    files1 = glob.glob(folder1)
    files2 = glob.glob(folder2)
    files3 = glob.glob(folder3)
    files4 = glob.glob(folder4)
    files5 = glob.glob(folder5)

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

    bhashTable = {}
    bcollision = []
    bcollisionLocation = []
    bcollisionCount = 0

    bbNose        = []
    bbNeck        = []
    bbRShoulder   = []
    bbRElbow      = []
    bbRWrist      = []
    bbLShoulder   = []
    bbLElbow      = []
    bbLWrist      = []
    bbRHip        = []
    bbRKnee       = []
    bbRAnkle      = []
    bbLHip        = []
    bbLKnee       = []
    bbLAnkle      = []
    bbREye        = []
    bbLEye        = []
    bbREar        = []
    bbLEar        = []

    bbhashTable = {}
    bbcollision = []
    bbcollisionLocation = []
    bbcollisionCount = 0

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

    chashTable = {}
    ccollision = []
    ccollisionLocation = []
    ccollisionCount = 0

    rNose        = []
    rNeck        = []
    rRShoulder   = []
    rRElbow      = []
    rRWrist      = []
    rLShoulder   = []
    rLElbow      = []
    rLWrist      = []
    rRHip        = []
    rRKnee       = []
    rRAnkle      = []
    rLHip        = []
    rLKnee       = []
    rLAnkle      = []
    rREye        = []
    rLEye        = []
    rREar        = []
    rLEar        = []

    rhashTable = {}
    rcollision = []
    rcollisionLocation = []
    rcollisionCount = 0

    tNose        = []
    tNeck        = []
    tRShoulder   = []
    tRElbow      = []
    tRWrist      = []
    tLShoulder   = []
    tLElbow      = []
    tLWrist      = []
    tRHip        = []
    tRKnee       = []
    tRAnkle      = []
    tLHip        = []
    tLKnee       = []
    tLAnkle      = []
    tREye        = []
    tLEye        = []
    tREar        = []
    tLEar        = []

    thashTable = {}
    tcollision = []
    tcollisionLocation = []
    tcollisionCount = 0

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
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in bhashTable:
                            bhashTable[current] += 1
                            bcollisionLocation.append(current)
                            bcollisionCount += 1

                        else:
                            bhashTable[current] = 0
            bhashTable = {}
            bcollision.append(bcollisionCount)
            bcollisionCount = 0

            if bcollision != 0:
                bcollisionLocation.append(current)

    bgetCount = np.array(bcollision)
    bTotalCount = sum(bgetCount)

    for file in files2:
        with open(file) as json_data:
            data = json.load(json_data)
            for parts in data['people']:
                bbNose.append(parts['pose_keypoints'][2])
                bbNeck.append(parts['pose_keypoints'][5])
                bbRShoulder.append(parts['pose_keypoints'][8])
                bbRElbow.append(parts['pose_keypoints'][11])
                bbRWrist.append(parts['pose_keypoints'][14])
                bbLShoulder.append(parts['pose_keypoints'][17])
                bbLElbow.append(parts['pose_keypoints'][20])
                bbLWrist.append(parts['pose_keypoints'][23])
                bbRHip.append(parts['pose_keypoints'][26])
                bbRKnee.append(parts['pose_keypoints'][29])
                bbRAnkle.append(parts['pose_keypoints'][32])
                bbLHip.append(parts['pose_keypoints'][35])
                bbLKnee.append(parts['pose_keypoints'][38])
                bbLAnkle.append(parts['pose_keypoints'][41])
                bbREye.append(parts['pose_keypoints'][44])
                bbLEye.append(parts['pose_keypoints'][47])
                bbREar.append(parts['pose_keypoints'][50])
                bbLEar.append(parts['pose_keypoints'][53])
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in bbhashTable:
                            bbhashTable[current] += 1
                            bbcollisionLocation.append(current)
                            bbcollisionCount += 1

                        else:
                            bbhashTable[current] = 0
            bbhashTable = {}
            bbcollision.append(bbcollisionCount)
            bbcollisionCount = 0

            if bbcollision != 0:
                bbcollisionLocation.append(current)

    bbgetCount = np.array(bbcollision)
    bbTotalCount = sum(bbgetCount)

    for file in files3:
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
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in chashTable:
                            chashTable[current] += 1
                            ccollisionLocation.append(current)
                            ccollisionCount += 1

                        else:
                            chashTable[current] = 0
            chashTable = {}
            ccollision.append(ccollisionCount)
            ccollisionCount = 0

            if ccollision != 0:
                ccollisionLocation.append(current)

    cgetCount = np.array(ccollision)
    cTotalCount = sum(cgetCount)

    for file in files4:
        with open(file) as json_data:
            data = json.load(json_data)
            for parts in data['people']:
                rNose.append(parts['pose_keypoints'][2])
                rNeck.append(parts['pose_keypoints'][5])
                rRShoulder.append(parts['pose_keypoints'][8])
                rRElbow.append(parts['pose_keypoints'][11])
                rRWrist.append(parts['pose_keypoints'][14])
                rLShoulder.append(parts['pose_keypoints'][17])
                rLElbow.append(parts['pose_keypoints'][20])
                rLWrist.append(parts['pose_keypoints'][23])
                rRHip.append(parts['pose_keypoints'][26])
                rRKnee.append(parts['pose_keypoints'][29])
                rRAnkle.append(parts['pose_keypoints'][32])
                rLHip.append(parts['pose_keypoints'][35])
                rLKnee.append(parts['pose_keypoints'][38])
                rLAnkle.append(parts['pose_keypoints'][41])
                rREye.append(parts['pose_keypoints'][44])
                rLEye.append(parts['pose_keypoints'][47])
                rREar.append(parts['pose_keypoints'][50])
                rLEar.append(parts['pose_keypoints'][53])
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in rhashTable:
                            rhashTable[current] += 1
                            rcollisionLocation.append(current)
                            rcollisionCount += 1

                        else:
                            rhashTable[current] = 0
            rhashTable = {}
            rcollision.append(rcollisionCount)
            rcollisionCount = 0

            if rcollision != 0:
                rcollisionLocation.append(current)

    rgetCount = np.array(rcollision)
    rTotalCount = sum(rgetCount)

    for file in files5:
        with open(file) as json_data:
            data = json.load(json_data)
            for parts in data['people']:
                tNose.append(parts['pose_keypoints'][2])
                tNeck.append(parts['pose_keypoints'][5])
                tRShoulder.append(parts['pose_keypoints'][8])
                tRElbow.append(parts['pose_keypoints'][11])
                tRWrist.append(parts['pose_keypoints'][14])
                tLShoulder.append(parts['pose_keypoints'][17])
                tLElbow.append(parts['pose_keypoints'][20])
                tLWrist.append(parts['pose_keypoints'][23])
                tRHip.append(parts['pose_keypoints'][26])
                tRKnee.append(parts['pose_keypoints'][29])
                tRAnkle.append(parts['pose_keypoints'][32])
                tLHip.append(parts['pose_keypoints'][35])
                tLKnee.append(parts['pose_keypoints'][38])
                tLAnkle.append(parts['pose_keypoints'][41])
                tREye.append(parts['pose_keypoints'][44])
                tLEye.append(parts['pose_keypoints'][47])
                tREar.append(parts['pose_keypoints'][50])
                tLEar.append(parts['pose_keypoints'][53])
                for i in range(0,56,3):
                    current = tuple(parts['pose_keypoints'][i:i+3])

                    if current != (0,0,0) and current != ():
                        if current in thashTable:
                            thashTable[current] += 1
                            tcollisionLocation.append(current)
                            tcollisionCount += 1

                        else:
                            thashTable[current] = 0
            thashTable = {}
            tcollision.append(tcollisionCount)
            tcollisionCount = 0

            if tcollision != 0:
                tcollisionLocation.append(current)

    tgetCount = np.array(tcollision)
    tTotalCount = sum(tgetCount)

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

    bbavgNose        = sum(bbNose)/len(bbNose)
    bbavgNeck        = sum(bbNeck)/len(bbNeck)
    bbavgRShoulder   = sum(bbRShoulder)/len(bbRShoulder)
    bbavgRElbow      = sum(bbRElbow)/len(bbRElbow)
    bbavgRWrist      = sum(bbRWrist)/len(bbRWrist)
    bbavgLShoulder   = sum(bbLShoulder)/len(bbLShoulder)
    bbavgLElbow      = sum(bbLElbow)/len(bbLElbow)
    bbavgLWrist      = sum(bbLWrist)/len(bbLWrist)
    bbavgRHip        = sum(bbRHip)/len(bbRHip)
    bbavgRKnee       = sum(bbRKnee)/len(bbRKnee)
    bbavgRAnkle      = sum(bbRAnkle)/len(bbRAnkle)
    bbavgLHip        = sum(bbLHip)/len(bbLHip)
    bbavgLKnee       = sum(bbLKnee)/len(bbLKnee)
    bbavgLAnkle      = sum(bbLAnkle)/len(bbLAnkle)
    bbavgREye        = sum(bbREye)/len(bbREye)
    bbavgLEye        = sum(bbLEye)/len(bbLEye)
    bbavgREar        = sum(bbREar)/len(bbREar)
    bbavgLEar        = sum(bbLEar)/len(bbLEar)

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

    ravgNose        = sum(rNose)/len(rNose)
    ravgNeck        = sum(rNeck)/len(rNeck)
    ravgRShoulder   = sum(rRShoulder)/len(rRShoulder)
    ravgRElbow      = sum(rRElbow)/len(rRElbow)
    ravgRWrist      = sum(rRWrist)/len(rRWrist)
    ravgLShoulder   = sum(rLShoulder)/len(rLShoulder)
    ravgLElbow      = sum(rLElbow)/len(rLElbow)
    ravgLWrist      = sum(rLWrist)/len(rLWrist)
    ravgRHip        = sum(rRHip)/len(rRHip)
    ravgRKnee       = sum(rRKnee)/len(rRKnee)
    ravgRAnkle      = sum(rRAnkle)/len(rRAnkle)
    ravgLHip        = sum(rLHip)/len(rLHip)
    ravgLKnee       = sum(rLKnee)/len(rLKnee)
    ravgLAnkle      = sum(rLAnkle)/len(rLAnkle)
    ravgREye        = sum(rREye)/len(rREye)
    ravgLEye        = sum(rLEye)/len(rLEye)
    ravgREar        = sum(rREar)/len(rREar)
    ravgLEar        = sum(rLEar)/len(rLEar)

    tavgNose        = sum(tNose)/len(tNose)
    tavgNeck        = sum(tNeck)/len(tNeck)
    tavgRShoulder   = sum(tRShoulder)/len(tRShoulder)
    tavgRElbow      = sum(tRElbow)/len(tRElbow)
    tavgRWrist      = sum(tRWrist)/len(tRWrist)
    tavgLShoulder   = sum(tLShoulder)/len(tLShoulder)
    tavgLElbow      = sum(tLElbow)/len(tLElbow)
    tavgLWrist      = sum(tLWrist)/len(tLWrist)
    tavgRHip        = sum(tRHip)/len(tRHip)
    tavgRKnee       = sum(tRKnee)/len(tRKnee)
    tavgRAnkle      = sum(tRAnkle)/len(tRAnkle)
    tavgLHip        = sum(tLHip)/len(tLHip)
    tavgLKnee       = sum(tLKnee)/len(tLKnee)
    tavgLAnkle      = sum(tLAnkle)/len(tLAnkle)
    tavgREye        = sum(tREye)/len(tREye)
    tavgLEye        = sum(tLEye)/len(tLEye)
    tavgREar        = sum(tREar)/len(tREar)
    tavgLEar        = sum(tLEar)/len(tLEar)

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
                                     'lear']).transpose()

    bbavgConf = pd.DataFrame({'nose': bavgNose,
                            'neck': bbavgNeck,
                            'rshoulder': bbavgRShoulder,
                            'relbow': bbavgRElbow,
                            'rwrist': bbavgRWrist,
                            'lshoulder': bbavgLShoulder,
                            'lelbow': bbavgLElbow,
                            'lwrist': bbavgLWrist,
                            'rhip': bbavgRHip,
                            'rknee': bbavgRKnee,
                            'rankle': bbavgRAnkle,
                            'lhip': bbavgLHip,
                            'lknee': bbavgLKnee,
                            'lankle': bbavgLAnkle,
                            'reye': bbavgREye,
                            'leye': bbavgLEye,
                            'rear': bbavgREar,
                            'lear': bbavgLEar},
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
                                     'lear']).transpose()

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
                                     'lear']).transpose()

    ravgConf = pd.DataFrame({'nose': bavgNose,
                            'neck': ravgNeck,
                            'rshoulder': ravgRShoulder,
                            'relbow': ravgRElbow,
                            'rwrist': ravgRWrist,
                            'lshoulder': ravgLShoulder,
                            'lelbow': ravgLElbow,
                            'lwrist': ravgLWrist,
                            'rhip': ravgRHip,
                            'rknee': ravgRKnee,
                            'rankle': ravgRAnkle,
                            'lhip': ravgLHip,
                            'lknee': ravgLKnee,
                            'lankle': ravgLAnkle,
                            'reye': ravgREye,
                            'leye': ravgLEye,
                            'rear': ravgREar,
                            'lear': ravgLEar},
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
                                     'lear']).transpose()

    tavgConf = pd.DataFrame({'nose': tavgNose,
                            'neck': tavgNeck,
                            'rshoulder': tavgRShoulder,
                            'relbow': tavgRElbow,
                            'rwrist': tavgRWrist,
                            'lshoulder': tavgLShoulder,
                            'lelbow': tavgLElbow,
                            'lwrist': tavgLWrist,
                            'rhip': tavgRHip,
                            'rknee': tavgRKnee,
                            'rankle': tavgRAnkle,
                            'lhip': tavgLHip,
                            'lknee': tavgLKnee,
                            'lankle': tavgLAnkle,
                            'reye': tavgREye,
                            'leye': tavgLEye,
                            'rear': tavgREar,
                            'lear': tavgLEar},
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
                                     'lear']).transpose()

    bavgConf = bavgConf.to_html()
    bbavgConf = bbavgConf.to_html()
    cavgConf = cavgConf.to_html()
    ravgConf = ravgConf.to_html()
    tavgConf = tavgConf.to_html()

    return render_template('compare.html', bavgConf = bavgConf, bbavgConf = bbavgConf, cavgConf = cavgConf, ravgConf = ravgConf, tavgConf = tavgConf, bTotalCount = bTotalCount, bbTotalCount = bbTotalCount, cTotalCount = cTotalCount, rTotalCount = rTotalCount, tTotalCount = tTotalCount)
