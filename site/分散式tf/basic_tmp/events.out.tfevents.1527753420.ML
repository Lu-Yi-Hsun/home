       �K"	   ����Abrain.Event:2���v      F^�	wC/����A"�

s
aConst"/job:worker/task:0/device:CPU:0*
valueB"  �?  �@*
dtype0*
_output_shapes
:
�
b/initial_valueConst"/job:worker/task:0/device:CPU:0*
dtype0*
_output_shapes
:*
valueB"  �?��L@
�
b
VariableV2"/job:worker/task:0/device:CPU:0*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
b/AssignAssignbb/initial_value"/job:worker/task:0/device:CPU:0*
_class

loc:@b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
b/readIdentityb"/job:worker/task:0/device:CPU:0*
_output_shapes
:*
T0*
_class

loc:@b
[
mulMulab/read"/job:worker/task:0/device:CPU:0*
T0*
_output_shapes
:
c
truedivRealDivab/read"/job:worker/task:0/device:CPU:0*
T0*
_output_shapes
:
^
addAddmultruediv"/job:worker/task:0/device:CPU:0*
T0*
_output_shapes
:
Z
mul_1Muladda"/job:worker/task:0/device:CPU:0*
T0*
_output_shapes
:
�
AssignAssignbmul_1"/job:worker/task:0/device:CPU:0*
use_locking(*
T0*
_class

loc:@b*
validate_shape(*
_output_shapes
:
8
initNoOp	^b/Assign"/job:worker/task:0/device:CPU:0"����      ���-	�u/����AJ�
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.8.02b'v1.8.0-2204-g717aa746e7'�

s
aConst"/job:worker/task:0/device:CPU:0*
valueB"  �?  �@*
dtype0*
_output_shapes
:
�
b/initial_valueConst"/job:worker/task:0/device:CPU:0*
valueB"  �?��L@*
dtype0*
_output_shapes
:
�
b
VariableV2"/job:worker/task:0/device:CPU:0*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
b/AssignAssignbb/initial_value"/job:worker/task:0/device:CPU:0*
use_locking(*
T0*
_class

loc:@b*
validate_shape(*
_output_shapes
:
q
b/readIdentityb"/job:worker/task:0/device:CPU:0*
T0*
_class

loc:@b*
_output_shapes
:
[
mulMulab/read"/job:worker/task:0/device:CPU:0*
T0*
_output_shapes
:
c
truedivRealDivab/read"/job:worker/task:0/device:CPU:0*
T0*
_output_shapes
:
^
addAddmultruediv"/job:worker/task:0/device:CPU:0*
_output_shapes
:*
T0
Z
mul_1Muladda"/job:worker/task:0/device:CPU:0*
T0*
_output_shapes
:
�
AssignAssignbmul_1"/job:worker/task:0/device:CPU:0*
T0*
_class

loc:@b*
validate_shape(*
_output_shapes
:*
use_locking(
8
initNoOp	^b/Assign"/job:worker/task:0/device:CPU:0""=
	variables0.
,
b:0b/Assignb/read:02b/initial_value:0"G
trainable_variables0.
,
b:0b/Assignb/read:02b/initial_value:0G��