# Convolutional Neural Network From Scratch
This repository uses NumPy to develop CNN. No Deep Learning Framework used. Used Keras just to get Cifar-10 dataset to test things out.

## Dependencies
1. NumPy
2. Tqdm
3. Keras (For Cifar-10 dataset only)

## Usage
`python main.py`

## Architecture In main.py

| Layer Type 	| in shape    	| out shape   	| params 	|
|------------	|-------------	|-------------	|--------	|
| input      	| (32, 32, 3) 	| (32, 32, 3) 	| 0      	|
| conv       	| (32, 32, 3) 	| (16, 16, 4) 	| 112    	|
| act        	| (16, 16, 4) 	| (16, 16, 4) 	| 0      	|
| flatten    	| (16, 16, 4) 	| 1024        	| 0      	|
| dense      	| 1024        	| 10          	| 10250  	|
| act        	| 10          	| 10          	| 0      	|


---------------------------------------------------------------------------<br/>
Total Number of Params: 10362<br/>
---------------------------------------------------------------------------<br/>