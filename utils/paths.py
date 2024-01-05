"""just a useful list of directories"""
import os


home_dir = os.environ['HOME']
debussy_PATH = os.path.join(home_dir, 'DEBUSSY_V2.2_2019/bin/')
noisy_PATH = os.path.join(home_dir, 'noisy/bin/noisy')

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, 'DATA')
templates_dir = os.path.join(project_dir, 'templates')
solvents_dir = os.path.join(project_dir, 'solvents')
