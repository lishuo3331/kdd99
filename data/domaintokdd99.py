import csv
import subprocess,multiprocessing
import sys
import pandas as pd
import time
import pymysql
import pickle

import numpy as np
import pandas as pd
import pickle
def map2major5(df):
    d = {
        'normal.': 0,
        'ipsweep.': 1,
        'mscan.': 1,
        'nmap.': 1,
        'portsweep.': 1,
        'saint.': 1,
        'satan.': 1,
        'apache2.': 1,
        'back.': 1,
        'mailbomb.': 1,
        'neptune.': 1,
        'pod.': 1,
        'land.': 1,
        'processtable.': 1,
        'smurf.': 1,
        'teardrop.': 1,
        'udpstorm.': 1,
        'buffer_overflow.': 1,
        'loadmodule.': 1,
        'perl.': 1,
        'ps.': 1,
        'rootkit.': 1,
        'sqlattack.': 1,
        'xterm.': 1,
        'ftp_write.': 1,
        'guess_passwd.': 1,
        'httptunnel.': 1,  # disputation resolved
        'imap.': 1,
        'multihop.': 1,  # disputation resolved
        'named.': 1,
        'phf.': 1,
        'sendmail.': 1,
        'snmpgetattack.': 1,
        'snmpguess.': 1,
        'worm.': 1,
        'xlock.': 1,
        'xsnoop.': 1,
        'spy.': 1,
        'warezclient.': 1,
        'warezmaster.': 1  # disputation resolved
    }

    # d = {
    #     'normal.': 0,
    #     'ipsweep.': 1,
    #     'mscan.': 1,
    #     'nmap.': 1,
    #     'portsweep.': 1,
    #     'saint.': 1,
    #     'satan.': 1,
    #     'apache2.': 2,
    #     'back.': 2,
    #     'mailbomb.': 2,
    #     'neptune.': 2,
    #     'pod.': 2,
    #     'land.': 2,
    #     'processtable.': 2,
    #     'smurf.': 2,
    #     'teardrop.': 2,
    #     'udpstorm.': 2,
    #     'buffer_overflow.': 3,
    #     'loadmodule.': 3,
    #     'perl.': 3,
    #     'ps.': 3,
    #     'rootkit.': 3,
    #     'sqlattack.': 3,
    #     'xterm.': 3,
    #     'ftp_write.': 4,
    #     'guess_passwd.': 4,
    #     'httptunnel.': 3,  # disputation resolved
    #     'imap.': 4,
    #     'multihop.': 4,  # disputation resolved
    #     'named.': 4,
    #     'phf.': 4,
    #     'sendmail.': 4,
    #     'snmpgetattack.': 4,
    #     'snmpguess.': 4,
    #     'worm.': 4,
    #     'xlock.': 4,
    #     'xsnoop.': 4,
    #     'spy.': 4,
    #     'warezclient.': 4,
    #     'warezmaster.': 4  # disputation resolved
    # }

    l = []
    for val in df['attack_type']:
        l.append(d[val])
    tmp_df = pd.DataFrame(l, columns=['attack_type'])
    df = df.drop('attack_type', axis=1)
    df = df.join(tmp_df)
    return df


def one_hot(df):
    service_one_hot = pd.get_dummies(df["service"])
    df = df.drop('service', axis=1)
    df = df.join(service_one_hot)
    # test data has this column in service, clashes with protocol_type
    # and not seen in training data, won't be learn by the model, safely delete
    if 'icmp' in df.columns:
        df = df.drop('icmp', axis=1)

    protocol_type_one_hot = pd.get_dummies(df["protocol_type"])
    df = df.drop('protocol_type', axis=1)
    df = df.join(protocol_type_one_hot)

    flag_type_one_hot = pd.get_dummies(df["flag"])
    df = df.drop('flag', axis=1)
    df = df.join(flag_type_one_hot)
    return df


def merge_sparse_feature(df):
    df.loc[(df['service'] == 'ntp_u')
           | (df['service'] == 'urh_i')
           | (df['service'] == 'tftp_u')
           | (df['service'] == 'red_i')
    , 'service'] = 'normal_service_group'

    df.loc[(df['service'] == 'pm_dump')
           | (df['service'] == 'http_2784')
           | (df['service'] == 'harvest')
           | (df['service'] == 'aol')
           | (df['service'] == 'http_8001')
    , 'service'] = 'satan_service_group'
    return df

def make_test_df(path,id):
    __ATTR_NAMES = ("duration",  # length (number of seconds) of the conn's
                "protocol_type",  # symbolic, type of the protocol, e.g. tcp, udp, etc.
                "service",  # symbolic, network service on the destination, e.g., http, telnet, etc.
                "flag",  # symbolic, normal or error status of the conn
                "src_bytes",  # number of data bytes from source to destination
                "dst_bytes",  # number of data bytes from destination to source
                "land",  # symbolic, 1 if conn is from/to the same host/port; 0 otherwise
                "wrong_fragment",  # number of ''wrong'' fragments 
                "urgent",  # number of urgent packets
                # ----------
                # ----- Basic features of individual TCP conn's -----
                # ----------
                "hot",  # number of ''hot'' indicators
                "num_failed_logins",  # number of failed login attempts 
                "logged_in",  # symbolic, 1 if successfully logged in; 0 otherwise
                "num_compromised",  # number of ''compromised'' conditions 
                "root_shell",  # 1 if root shell is obtained; 0 otherwise 
                "su_attempted",  # 1 if ''su root'' command attempted; 0 otherwise 
                "num_root",  # number of ''root'' accesses 
                "num_file_creations",  # number of file creation operations
                "num_shells",  # number of shell prompts 
                "num_access_files",  # number of operations on access control files
                "num_outbound_cmds",  # number of outbound commands in an ftp session 
                "is_host_login",  # symbolic, 1 if the login belongs to the ''hot'' list; 0 otherwise 
                "is_guest_login",  # symbolic, 1 if the login is a ''guest''login; 0 otherwise 
                # ----------
                # ----- Content features within a conn suggested by domain knowledge -----
                # ----------
                "count",  # number of conn's to the same host as the current conn in the past two seconds 
                # Time-based Traffic Features (examine only the conn in the past two seconds):
                # 1. Same Host, have the same destination host as the current conn
                # 2. Same Service, have the same service as the current conn.
                "srv_count",  # SH, number of conn's to the same service as the current conn
                "serror_rate",  # SH, % of conn's that have SYN errors
                "srv_serror_rate",  # SS, % of conn's that have SYN errors
                "rerror_rate",  # SH, % of conn's that have REJ errors 
                "srv_rerror_rate",  # SS, % of conn's that have REJ errors 
                "same_srv_rate",  # SH, % of conn's to the same service 
                "diff_srv_rate",  # SH, % of conn's to different services 
                "srv_diff_host_rate",  # SH,  % of conn's to different hosts 
                # ----------
                # Host-base Traffic Features, constructed using a window of 100 conn's to the same host
                "dst_host_count",
                "dst_host_srv_count",
                "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate",
                "dst_host_srv_serror_rate",
                "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate",
                # ----------
                # category
                "attack_type"
                    )

    # df = pd.read_csv(r'../data/4.csv', header=None, names=__ATTR_NAMES)
    df = pd.read_csv(path, header=None, names=__ATTR_NAMES)

    df.services = df.service.unique()
    df.flags = df.flag.unique()
    df.protocol_types = df.protocol_type.unique()
    df.attack_types = df.attack_type.unique()

    df = merge_sparse_feature(df)  #合并稀疏特征
    df = one_hot(df)#独热编码

    with open(r'../data/selected_feat_names.pkl', 'rb') as f:
        selected_feat_names = pickle.load(f)
    for i in selected_feat_names:
        if i in df:
            print('ok')
        else:
            print(i)
            df[i] = '0'
    df.to_csv(path,header=None)
    # df = map2major5(df)#合并攻击类型

# percentage check, to make sure the mapping is correct
# print(df[df['attack_type'] == 0].shape[0] / df.shape[0])
# print(df[df['attack_type'] == 1].shape[0] / df.shape[0])
# print(df[df['attack_type'] == 2].shape[0] / df.shape[0])
# print(df[df['attack_type'] == 3].shape[0] / df.shape[0])
# print(df[df['attack_type'] == 4].shape[0] / df.shape[0])
#     f = open('.pkl'.format(id), 'wb')
    with open('{}.pkl'.format(id), 'wb') as f:
        pickle.dump(df, f)

def create_csv(path):
    with open(path,'wb') as f:
        csv_write = csv.writer(f)
def create_pkl(path):
    with open(path,'wb') as f:
        csv_write = csv.writer(f)
def write_csv(path,data):
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = data
        csv_write.writerow(data_row)


def RunShellWithReturnCode(command,print_output=True,universal_newlines=True,id=0):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=universal_newlines)
    n = 0
    if print_output:
        output_array = []
        path = '{}.csv'.format(id)
        create_csv(path)
        while True:
            # with open(r'../data/selected_feat_names.pkl', 'rb') as f:
            #     selected_feat_names = pickle.load(f)
            line = p.stdout.readline().strip()
            if not line:
                break
            list1=line.split(',')
            print(list1)
            for i in range (0,13):
                list1.insert(9,'0')
            print(list1)
            str1=''.join(list1)#list to string
            print(str1)
            n = n+1
            write_csv(path,list1)
            print(n)
            if(n==1):
               make_test_df(path, id)
               with open('{}.pkl'.format(id), 'rb') as f:
                   df = pickle.load(f)
               with open(r'../data/selected_feat_names.pkl', 'rb') as f:
                   selected_feat_names = pickle.load(f)
               print(df)
               X = df[selected_feat_names].values
               print("data loaded")

               time.sleep(1)
               n=0
            # print (line.strip("/n"))
            # print(line)
            output_array.append(line)
        output ="".join(output_array)
    else:
        output = p.stdout.read()
    p.wait()
    errout = p.stderr.read()
    if print_output and errout:
        print >> sys.stderr, errout
    p.stdout.close()
    p.stderr.close()
    return output, p.returncode

p1 = multiprocessing.Process(target=RunShellWithReturnCode,args=('cd /home && cd .. &&./home/msi/Downloads/kdd99_feature_extractor-master/build-files/src/kdd99extractor -i 1',True,True,1))
# p4 = multiprocessing.Process(target=RunShellWithReturnCode,args=('cd /home && cd .. &&./home/msi/Downloads/kdd99_feature_extractor-master/build-files/src/kdd99extractor -i 4',True,True,4))
# p5 = multiprocessing.Process(target=RunShellWithReturnCode,args=('cd /home && cd .. &&./home/msi/Downloads/kdd99_feature_extractor-master/build-files/src/kdd99extractor -i 5',True,True,5))
# p6 = multiprocessing.Process(target=RunShellWithReturnCode,args=('cd /home && cd .. &&./home/msi/Downloads/kdd99_feature_extractor-master/build-files/src/kdd99extractor -i 6',True,True,6))

p1.start()
# p4.start()
# p5.start()
# p6.start()
print("The number of CPU is:" + str(multiprocessing.cpu_count()))
for p in multiprocessing.active_children():
    print("child   p.name:" + p.name + "\tp.id" + str(p.pid))

# RunShellWithReturnCode('cd /home && cd .. &&./home/msi/Downloads/kdd99_feature_extractor-master/build-files/src/kdd99extractor -i 4')# centos7.0
# RunShellWithReturnCode('cd /home && cd .. &&./home/msi/Downloads/kdd99_feature_extractor-master/build-files/src/kdd99extractor -i 5')# ubuntu1804
# RunShellWithReturnCode('cd /home && cd .. &&./home/msi/Downloads/kdd99_feature_extractor-master/build-files/src/kdd99extractor -i 6')# win7
