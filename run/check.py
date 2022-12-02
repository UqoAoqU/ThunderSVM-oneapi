#!/usr/bin/python3
import sys
if len(sys.argv) != 3:
    print('Usage : python3 check.py <result_file> <refer_file>')
    quit()
lines = open(sys.argv[1]).readlines()
exec(lines[0])
obj
exec(lines[1])
rho
lines = open(sys.argv[2]).readlines()
exec(lines[0])
exec(lines[1])
exec(lines[2])

pass_ = True
if abs((obj-obj_refer)/obj_refer) > tol:
    print('obj      :', obj, '\nobj_refer:', obj_refer)
    pass_ = False
if abs((rho-rho_refer)/rho_refer) > tol:
    print('rho      :', rho, '\nrho_refer:', rho_refer)
    pass_ = False

if pass_:
    print('PASS')
else:
    print('!!! WRONG !!!')

