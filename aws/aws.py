import subprocess    
import time
import json
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ec2_json', default='ec2.json')
    parser.add_argument('--instance_json', default='instances.json')

    args = parser.parse_args()
    ec2_json = args.ec2_json
    instance_json = args.instance_json
    return instance_json, ec2_json


def create_instace(ec2_json):
    cmd_line = "aws ec2 run-instances --cli-input-json file://" + ec2_json
    p = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = str(p.communicate()[0], 'utf-8')

    print('Waiting for instance setup...')
    time.sleep(600)
    return out

def get_instance_info():
    cmd_line = "aws ec2 describe-instances --query Reservations[].Instances[]"
    p = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    info = str(p.communicate()[0], 'utf-8')

    return info

def parse_instance_info(info):
    # Parse for running instances
    instances = {}
    instance_details = json.loads(info)
    for i in range(len(instance_details)):
        if 'PublicIpAddress' in instance_details[i]:
            instances[instance_details[i]['LaunchTime']] = (instance_details[i]['PublicIpAddress'], instance_details[i]['InstanceId'])

    # Get instance timestamps
    timestamps = list(instances.keys())
    timestamps.sort()

    return instances, timestamps

def configure_instance(ec2_json, ip_address):
    with open(ec2_json, 'r') as f:
        ec2_config = json.load(f)

    ssh_cmd = "ssh -oStrictHostKeyChecking=no -Y -i ~/.ssh/" + ec2_config['KeyName'] + ".pem ubuntu@" + ip_address + " \" $(cat initialization.txt)\" "
    p = subprocess.Popen(ssh_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = str(p.communicate()[0], 'utf-8')
    
    #ssh_cmd1 = "ssh -oStrictHostKeyChecking=no -Y -i ~/.ssh/" + ec2_config['KeyName'] + ".pem ubuntu@" + ip_address
    #ssh_cmd2 = "'sudo apt-get update'"
    #ssh_cmd = ssh_cmd1 + ' ' + ssh_cmd2
    #ssh_cmd = "ssh -Y -i ~/.ssh/" + ec2_config['KeyName'] + ".pem ubuntu@" + ip_address + " 'sudo apt-get update'"
    #print(ssh_cmd)
    
    return out

def main():
    # Parse input arguments
    instance_json, ec2_json = parse_args()
    
    # Create AWS instance
    out = create_instace(ec2_json)
    print(out)

    # Get instance info, and parse for running instances
    info = get_instance_info()
    instances, timestamps = parse_instance_info(info)

    # Save parsed info as JSON
    with open(instance_json, 'w') as f:
        json.dump(instances, f)

    # Configure instance just created
    latest_instance = instances[timestamps[-1]]
    ip_address = latest_instance[0]

    out = configure_instance(ec2_json, ip_address)
    print(out)

if __name__ == "__main__":
    main()