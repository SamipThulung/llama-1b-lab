"""
Transformer models:
Size    | d_model |  d_ff  | num_layers | num_heads |
-----------------------------------------------------
small   |   768   |  3072  |     12     | 12        |
medium  |   1024  |  4096  |     24     | 16        |
large   |   1280  |  5120  |     36     | 20        |
xl      |   2560  |  10240 |     32     | 32        |
-----------------------------------------------------
"""

S_MODEL = {
	'd_model' : 768,
	'd_ff' : 3072,
	'num_layers' : 12,
	'num_heads' : 12  
}
M_MODEL = {
	'd_model' : 1024,
	'd_ff' : 4096,
	'num_layers' : 24,
	'num_heads' : 16  
}
L_MODEL = {
	'd_model' : 1280,
	'd_ff' : 5120,
	'num_layers' : 36,
	'num_heads' : 20  
}
XL_MODEL = {
	'd_model' : 2560,
	'd_ff' : 10240,
	'num_layers' : 32,
	'num_heads' : 32  
}