#csv format: DateTime | Bid | Ask | Volume
#data_path = 'E:/OS_Share/008-FinTech/dataset/20200703_20200505_EUR2USD.csv'
#data_path = 'E:/OS_Share/008-FinTech/dataset/test2_EUR2USD.csv'
data_path = 'E:/OS_Share/008-FinTech/dataset/test3_EUR2USD.csv'
#Raw data from csv
raw_data = []
#Only contain bid data (1d np.array) 
bid_data = []
#A list to store data order (1d np.array)
time_line = []
#Total number of data
total_data = 0


#Shuffled and normalized y_data
y_data_ready = []
bid_data_ready = []

#Training and testing dataset
x_train = []
x_test = []
y_train = []
y_test = []

training_fraction = 0.75
epoches = 2000
batch_size =1000
initial_learning_rate = 0.01
decay_step = 100
decay_factor = 0.96
