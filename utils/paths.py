"""just a useful list of directories"""
import os


home_dir = os.environ['HOME']
debussy_PATH = os.path.join(home_dir, 'DEBUSSY_V2.2_2019/bin/')
noisy_PATH = os.path.join(home_dir, 'noisy/noisymonocol')

# project_dir = '/home/lucia/ML_PhD_projects/Dimension_recognition'
# project_dir = os.path.dirname(os.getcwd())
project_dir = os.getcwd()
data_dir = os.path.join(project_dir, 'DATA')
templates_dir = os.path.join(project_dir, 'templates')
solvents_dir = os.path.join(project_dir, 'solvents')

# !! if I import this somewhere, it runs there, so either:
# - explicit project_dir
# OR
# - make it somehow run in Dimension_recognition if I want a relative path for whoever installs it. <--NOW

