       ЃK"	  Р5nзAbrain.Event:2EьXеJі     њkЏ	fФш5nзA"Нь

q_eval/statesPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџTT*$
shape:џџџџџџџџџTT
v
q_eval/action_takenPlaceholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
i
q_eval/q_valuePlaceholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
[
q_eval/sequence_lengthPlaceholder*
dtype0*
_output_shapes
:*
shape:
V
q_eval/batch_sizePlaceholder*
dtype0*
_output_shapes
:*
shape:
v
q_eval/cell_statePlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
s
q_eval/h_statePlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
X
q_eval/Reward/Time_stepPlaceholder*
dtype0*
_output_shapes
: *
shape: 
x
q_eval/Reward/Time_step_1/tagsConst**
value!B Bq_eval/Reward/Time_step_1*
dtype0*
_output_shapes
: 

q_eval/Reward/Time_step_1ScalarSummaryq_eval/Reward/Time_step_1/tagsq_eval/Reward/Time_step*
T0*
_output_shapes
: 
b
!q_eval/TotalWaitingTime/Time_stepPlaceholder*
dtype0*
_output_shapes
: *
shape: 

(q_eval/TotalWaitingTime/Time_step_1/tagsConst*
dtype0*
_output_shapes
: *4
value+B) B#q_eval/TotalWaitingTime/Time_step_1
Ђ
#q_eval/TotalWaitingTime/Time_step_1ScalarSummary(q_eval/TotalWaitingTime/Time_step_1/tags!q_eval/TotalWaitingTime/Time_step*
_output_shapes
: *
T0
\
q_eval/TotalDelay/Time_stepPlaceholder*
dtype0*
_output_shapes
: *
shape: 

"q_eval/TotalDelay/Time_step_1/tagsConst*.
value%B# Bq_eval/TotalDelay/Time_step_1*
dtype0*
_output_shapes
: 

q_eval/TotalDelay/Time_step_1ScalarSummary"q_eval/TotalDelay/Time_step_1/tagsq_eval/TotalDelay/Time_step*
T0*
_output_shapes
: 
З
6q_eval/conv1/kernel/Initializer/truncated_normal/shapeConst*&
_class
loc:@q_eval/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ђ
5q_eval/conv1/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_eval/conv1/kernel*
valueB
 *    
Є
7q_eval/conv1/kernel/Initializer/truncated_normal/stddevConst*&
_class
loc:@q_eval/conv1/kernel*
valueB
 *аdN>*
dtype0*
_output_shapes
: 

@q_eval/conv1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6q_eval/conv1/kernel/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:*

seed *
T0*&
_class
loc:@q_eval/conv1/kernel*
seed2 

4q_eval/conv1/kernel/Initializer/truncated_normal/mulMul@q_eval/conv1/kernel/Initializer/truncated_normal/TruncatedNormal7q_eval/conv1/kernel/Initializer/truncated_normal/stddev*
T0*&
_class
loc:@q_eval/conv1/kernel*&
_output_shapes
:
§
0q_eval/conv1/kernel/Initializer/truncated_normalAdd4q_eval/conv1/kernel/Initializer/truncated_normal/mul5q_eval/conv1/kernel/Initializer/truncated_normal/mean*&
_output_shapes
:*
T0*&
_class
loc:@q_eval/conv1/kernel
П
q_eval/conv1/kernel
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *&
_class
loc:@q_eval/conv1/kernel*
	container 
э
q_eval/conv1/kernel/AssignAssignq_eval/conv1/kernel0q_eval/conv1/kernel/Initializer/truncated_normal*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:

q_eval/conv1/kernel/readIdentityq_eval/conv1/kernel*&
_output_shapes
:*
T0*&
_class
loc:@q_eval/conv1/kernel

#q_eval/conv1/bias/Initializer/ConstConst*$
_class
loc:@q_eval/conv1/bias*
valueB*
з#<*
dtype0*
_output_shapes
:
Ѓ
q_eval/conv1/bias
VariableV2*$
_class
loc:@q_eval/conv1/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ю
q_eval/conv1/bias/AssignAssignq_eval/conv1/bias#q_eval/conv1/bias/Initializer/Const*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:*
use_locking(

q_eval/conv1/bias/readIdentityq_eval/conv1/bias*
T0*$
_class
loc:@q_eval/conv1/bias*
_output_shapes
:
k
q_eval/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
я
q_eval/conv1/Conv2DConv2Dq_eval/statesq_eval/conv1/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ*
	dilations


q_eval/conv1/BiasAddBiasAddq_eval/conv1/Conv2Dq_eval/conv1/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ*
T0
c
q_eval/ReluReluq_eval/conv1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
З
6q_eval/conv2/kernel/Initializer/truncated_normal/shapeConst*&
_class
loc:@q_eval/conv2/kernel*%
valueB"             *
dtype0*
_output_shapes
:
Ђ
5q_eval/conv2/kernel/Initializer/truncated_normal/meanConst*&
_class
loc:@q_eval/conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
7q_eval/conv2/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_eval/conv2/kernel*
valueB
 *аdЮ=

@q_eval/conv2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6q_eval/conv2/kernel/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
: *

seed *
T0*&
_class
loc:@q_eval/conv2/kernel*
seed2 

4q_eval/conv2/kernel/Initializer/truncated_normal/mulMul@q_eval/conv2/kernel/Initializer/truncated_normal/TruncatedNormal7q_eval/conv2/kernel/Initializer/truncated_normal/stddev*
T0*&
_class
loc:@q_eval/conv2/kernel*&
_output_shapes
: 
§
0q_eval/conv2/kernel/Initializer/truncated_normalAdd4q_eval/conv2/kernel/Initializer/truncated_normal/mul5q_eval/conv2/kernel/Initializer/truncated_normal/mean*
T0*&
_class
loc:@q_eval/conv2/kernel*&
_output_shapes
: 
П
q_eval/conv2/kernel
VariableV2*
shape: *
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_eval/conv2/kernel*
	container 
э
q_eval/conv2/kernel/AssignAssignq_eval/conv2/kernel0q_eval/conv2/kernel/Initializer/truncated_normal*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(

q_eval/conv2/kernel/readIdentityq_eval/conv2/kernel*
T0*&
_class
loc:@q_eval/conv2/kernel*&
_output_shapes
: 

#q_eval/conv2/bias/Initializer/ConstConst*
dtype0*
_output_shapes
: *$
_class
loc:@q_eval/conv2/bias*
valueB *
з#<
Ѓ
q_eval/conv2/bias
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@q_eval/conv2/bias
Ю
q_eval/conv2/bias/AssignAssignq_eval/conv2/bias#q_eval/conv2/bias/Initializer/Const*
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: 

q_eval/conv2/bias/readIdentityq_eval/conv2/bias*
_output_shapes
: *
T0*$
_class
loc:@q_eval/conv2/bias
k
q_eval/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
э
q_eval/conv2/Conv2DConv2Dq_eval/Reluq_eval/conv2/kernel/read*/
_output_shapes
:џџџџџџџџџ		 *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

q_eval/conv2/BiasAddBiasAddq_eval/conv2/Conv2Dq_eval/conv2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ		 
e
q_eval/Relu_1Reluq_eval/conv2/BiasAdd*/
_output_shapes
:џџџџџџџџџ		 *
T0
e
q_eval/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ 
  

q_eval/ReshapeReshapeq_eval/Relu_1q_eval/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ 
[
q_eval/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value
B : 

q_eval/Reshape_1/shapePackq_eval/batch_sizeq_eval/sequence_lengthq_eval/Reshape_1/shape/2*
N*
_output_shapes
:*
T0*

axis 

q_eval/Reshape_1Reshapeq_eval/Reshapeq_eval/Reshape_1/shape*
T0*
Tshape0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ 
Q
q_eval/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
X
q_eval/rnn/range/startConst*
dtype0*
_output_shapes
: *
value	B :
X
q_eval/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_eval/rnn/rangeRangeq_eval/rnn/range/startq_eval/rnn/Rankq_eval/rnn/range/delta*
_output_shapes
:*

Tidx0
k
q_eval/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
X
q_eval/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

q_eval/rnn/concatConcatV2q_eval/rnn/concat/values_0q_eval/rnn/rangeq_eval/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:

q_eval/rnn/transpose	Transposeq_eval/Reshape_1q_eval/rnn/concat*
Tperm0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ 
a
q_eval/rnn/sequence_lengthIdentityq_eval/sequence_length*
T0*
_output_shapes
:
d
q_eval/rnn/ShapeShapeq_eval/rnn/transpose*
T0*
out_type0*
_output_shapes
:
h
q_eval/rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
j
 q_eval/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
j
 q_eval/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
А
q_eval/rnn/strided_sliceStridedSliceq_eval/rnn/Shapeq_eval/rnn/strided_slice/stack q_eval/rnn/strided_slice/stack_1 q_eval/rnn/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
u
q_eval/rnn/Shape_1Shapeq_eval/rnn/sequence_length*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
l
q_eval/rnn/stackPackq_eval/rnn/strided_slice*
N*
_output_shapes
:*
T0*

axis 
m
q_eval/rnn/EqualEqualq_eval/rnn/Shape_1q_eval/rnn/stack*
T0*#
_output_shapes
:џџџџџџџџџ
Z
q_eval/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
n
q_eval/rnn/AllAllq_eval/rnn/Equalq_eval/rnn/Const*
_output_shapes
: *
	keep_dims( *

Tidx0

q_eval/rnn/Assert/ConstConst*K
valueBB@ B:Expected shape for Tensor q_eval/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
j
q_eval/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

q_eval/rnn/Assert/Assert/data_0Const*K
valueBB@ B:Expected shape for Tensor q_eval/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
p
q_eval/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
К
q_eval/rnn/Assert/AssertAssertq_eval/rnn/Allq_eval/rnn/Assert/Assert/data_0q_eval/rnn/stackq_eval/rnn/Assert/Assert/data_2q_eval/rnn/Shape_1*
T
2*
	summarize
|
q_eval/rnn/CheckSeqLenIdentityq_eval/rnn/sequence_length^q_eval/rnn/Assert/Assert*
T0*
_output_shapes
:
f
q_eval/rnn/Shape_2Shapeq_eval/rnn/transpose*
_output_shapes
:*
T0*
out_type0
j
 q_eval/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
l
"q_eval/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"q_eval/rnn/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
К
q_eval/rnn/strided_slice_1StridedSliceq_eval/rnn/Shape_2 q_eval/rnn/strided_slice_1/stack"q_eval/rnn/strided_slice_1/stack_1"q_eval/rnn/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
f
q_eval/rnn/Shape_3Shapeq_eval/rnn/transpose*
T0*
out_type0*
_output_shapes
:
j
 q_eval/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
l
"q_eval/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"q_eval/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
К
q_eval/rnn/strided_slice_2StridedSliceq_eval/rnn/Shape_3 q_eval/rnn/strided_slice_2/stack"q_eval/rnn/strided_slice_2/stack_1"q_eval/rnn/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
[
q_eval/rnn/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 

q_eval/rnn/ExpandDims
ExpandDimsq_eval/rnn/strided_slice_2q_eval/rnn/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
]
q_eval/rnn/Const_1Const*
dtype0*
_output_shapes
:*
valueB:
Z
q_eval/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

q_eval/rnn/concat_1ConcatV2q_eval/rnn/ExpandDimsq_eval/rnn/Const_1q_eval/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
[
q_eval/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

q_eval/rnn/zerosFillq_eval/rnn/concat_1q_eval/rnn/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
R
q_eval/rnn/Rank_1Rankq_eval/rnn/CheckSeqLen*
T0*
_output_shapes
: 
Z
q_eval/rnn/range_1/startConst*
dtype0*
_output_shapes
: *
value	B : 
Z
q_eval/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_eval/rnn/range_1Rangeq_eval/rnn/range_1/startq_eval/rnn/Rank_1q_eval/rnn/range_1/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ

q_eval/rnn/MinMinq_eval/rnn/CheckSeqLenq_eval/rnn/range_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
q_eval/rnn/Rank_2Rankq_eval/rnn/CheckSeqLen*
_output_shapes
: *
T0
Z
q_eval/rnn/range_2/startConst*
dtype0*
_output_shapes
: *
value	B : 
Z
q_eval/rnn/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_eval/rnn/range_2Rangeq_eval/rnn/range_2/startq_eval/rnn/Rank_2q_eval/rnn/range_2/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ

q_eval/rnn/MaxMaxq_eval/rnn/CheckSeqLenq_eval/rnn/range_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Q
q_eval/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 

q_eval/rnn/TensorArrayTensorArrayV3q_eval/rnn/strided_slice_1*%
element_shape:џџџџџџџџџ*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*6
tensor_array_name!q_eval/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 

q_eval/rnn/TensorArray_1TensorArrayV3q_eval/rnn/strided_slice_1*5
tensor_array_name q_eval/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *%
element_shape:џџџџџџџџџ *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
w
#q_eval/rnn/TensorArrayUnstack/ShapeShapeq_eval/rnn/transpose*
T0*
out_type0*
_output_shapes
:
{
1q_eval/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
}
3q_eval/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
}
3q_eval/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

+q_eval/rnn/TensorArrayUnstack/strided_sliceStridedSlice#q_eval/rnn/TensorArrayUnstack/Shape1q_eval/rnn/TensorArrayUnstack/strided_slice/stack3q_eval/rnn/TensorArrayUnstack/strided_slice/stack_13q_eval/rnn/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
k
)q_eval/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
k
)q_eval/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
р
#q_eval/rnn/TensorArrayUnstack/rangeRange)q_eval/rnn/TensorArrayUnstack/range/start+q_eval/rnn/TensorArrayUnstack/strided_slice)q_eval/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ

Eq_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3q_eval/rnn/TensorArray_1#q_eval/rnn/TensorArrayUnstack/rangeq_eval/rnn/transposeq_eval/rnn/TensorArray_1:1*
T0*'
_class
loc:@q_eval/rnn/transpose*
_output_shapes
: 
V
q_eval/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
d
q_eval/rnn/MaximumMaximumq_eval/rnn/Maximum/xq_eval/rnn/Max*
T0*
_output_shapes
: 
n
q_eval/rnn/MinimumMinimumq_eval/rnn/strided_slice_1q_eval/rnn/Maximum*
T0*
_output_shapes
: 
d
"q_eval/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Т
q_eval/rnn/while/EnterEnter"q_eval/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
Б
q_eval/rnn/while/Enter_1Enterq_eval/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
К
q_eval/rnn/while/Enter_2Enterq_eval/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
Х
q_eval/rnn/while/Enter_3Enterq_eval/cell_state*
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant( 
Т
q_eval/rnn/while/Enter_4Enterq_eval/h_state*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*.

frame_name q_eval/rnn/while/while_context

q_eval/rnn/while/MergeMergeq_eval/rnn/while/Enterq_eval/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 

q_eval/rnn/while/Merge_1Mergeq_eval/rnn/while/Enter_1 q_eval/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 

q_eval/rnn/while/Merge_2Mergeq_eval/rnn/while/Enter_2 q_eval/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

q_eval/rnn/while/Merge_3Mergeq_eval/rnn/while/Enter_3 q_eval/rnn/while/NextIteration_3*
T0*
N**
_output_shapes
:џџџџџџџџџ: 

q_eval/rnn/while/Merge_4Mergeq_eval/rnn/while/Enter_4 q_eval/rnn/while/NextIteration_4*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
s
q_eval/rnn/while/LessLessq_eval/rnn/while/Mergeq_eval/rnn/while/Less/Enter*
T0*
_output_shapes
: 
П
q_eval/rnn/while/Less/EnterEnterq_eval/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
y
q_eval/rnn/while/Less_1Lessq_eval/rnn/while/Merge_1q_eval/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
Й
q_eval/rnn/while/Less_1/EnterEnterq_eval/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
q
q_eval/rnn/while/LogicalAnd
LogicalAndq_eval/rnn/while/Lessq_eval/rnn/while/Less_1*
_output_shapes
: 
Z
q_eval/rnn/while/LoopCondLoopCondq_eval/rnn/while/LogicalAnd*
_output_shapes
: 
Ђ
q_eval/rnn/while/SwitchSwitchq_eval/rnn/while/Mergeq_eval/rnn/while/LoopCond*
_output_shapes
: : *
T0*)
_class
loc:@q_eval/rnn/while/Merge
Ј
q_eval/rnn/while/Switch_1Switchq_eval/rnn/while/Merge_1q_eval/rnn/while/LoopCond*
T0*+
_class!
loc:@q_eval/rnn/while/Merge_1*
_output_shapes
: : 
Ј
q_eval/rnn/while/Switch_2Switchq_eval/rnn/while/Merge_2q_eval/rnn/while/LoopCond*
T0*+
_class!
loc:@q_eval/rnn/while/Merge_2*
_output_shapes
: : 
Ь
q_eval/rnn/while/Switch_3Switchq_eval/rnn/while/Merge_3q_eval/rnn/while/LoopCond*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*+
_class!
loc:@q_eval/rnn/while/Merge_3
Ь
q_eval/rnn/while/Switch_4Switchq_eval/rnn/while/Merge_4q_eval/rnn/while/LoopCond*
T0*+
_class!
loc:@q_eval/rnn/while/Merge_4*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
a
q_eval/rnn/while/IdentityIdentityq_eval/rnn/while/Switch:1*
_output_shapes
: *
T0
e
q_eval/rnn/while/Identity_1Identityq_eval/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
e
q_eval/rnn/while/Identity_2Identityq_eval/rnn/while/Switch_2:1*
_output_shapes
: *
T0
w
q_eval/rnn/while/Identity_3Identityq_eval/rnn/while/Switch_3:1*
T0*(
_output_shapes
:џџџџџџџџџ
w
q_eval/rnn/while/Identity_4Identityq_eval/rnn/while/Switch_4:1*(
_output_shapes
:џџџџџџџџџ*
T0
t
q_eval/rnn/while/add/yConst^q_eval/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
q_eval/rnn/while/addAddq_eval/rnn/while/Identityq_eval/rnn/while/add/y*
T0*
_output_shapes
: 
с
"q_eval/rnn/while/TensorArrayReadV3TensorArrayReadV3(q_eval/rnn/while/TensorArrayReadV3/Enterq_eval/rnn/while/Identity_1*q_eval/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:џџџџџџџџџ 
Ю
(q_eval/rnn/while/TensorArrayReadV3/EnterEnterq_eval/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
љ
*q_eval/rnn/while/TensorArrayReadV3/Enter_1EnterEq_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context

q_eval/rnn/while/GreaterEqualGreaterEqualq_eval/rnn/while/Identity_1#q_eval/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes
:
Х
#q_eval/rnn/while/GreaterEqual/EnterEnterq_eval/rnn/CheckSeqLen*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant(
Н
<q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB"      
Џ
:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB
 *њ<!Н*
dtype0*
_output_shapes
: 
Џ
:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB
 *њ<!=*
dtype0*
_output_shapes
: 

Dq_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform<q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*

seed *
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
seed2 *
dtype0* 
_output_shapes
:
 

:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/subSub:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/max:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
_output_shapes
: 

:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMulDq_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniform:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel* 
_output_shapes
:
 

6q_eval/rnn/lstm_cell/kernel/Initializer/random_uniformAdd:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/mul:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel* 
_output_shapes
:
 
У
q_eval/rnn/lstm_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
 *
shared_name *.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
	container *
shape:
 

"q_eval/rnn/lstm_cell/kernel/AssignAssignq_eval/rnn/lstm_cell/kernel6q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
 *
use_locking(*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel
t
 q_eval/rnn/lstm_cell/kernel/readIdentityq_eval/rnn/lstm_cell/kernel*
T0* 
_output_shapes
:
 
Д
;q_eval/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
Є
1q_eval/rnn/lstm_cell/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB
 *    

+q_eval/rnn/lstm_cell/bias/Initializer/zerosFill;q_eval/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensor1q_eval/rnn/lstm_cell/bias/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
Е
q_eval/rnn/lstm_cell/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
	container 
я
 q_eval/rnn/lstm_cell/bias/AssignAssignq_eval/rnn/lstm_cell/bias+q_eval/rnn/lstm_cell/bias/Initializer/zeros*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
k
q_eval/rnn/lstm_cell/bias/readIdentityq_eval/rnn/lstm_cell/bias*
_output_shapes	
:*
T0

&q_eval/rnn/while/lstm_cell/concat/axisConst^q_eval/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
о
!q_eval/rnn/while/lstm_cell/concatConcatV2"q_eval/rnn/while/TensorArrayReadV3q_eval/rnn/while/Identity_4&q_eval/rnn/while/lstm_cell/concat/axis*
T0*
N*(
_output_shapes
:џџџџџџџџџ *

Tidx0
а
!q_eval/rnn/while/lstm_cell/MatMulMatMul!q_eval/rnn/while/lstm_cell/concat'q_eval/rnn/while/lstm_cell/MatMul/Enter*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
л
'q_eval/rnn/while/lstm_cell/MatMul/EnterEnter q_eval/rnn/lstm_cell/kernel/read*
parallel_iterations * 
_output_shapes
:
 *.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant(
Ф
"q_eval/rnn/while/lstm_cell/BiasAddBiasAdd!q_eval/rnn/while/lstm_cell/MatMul(q_eval/rnn/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
е
(q_eval/rnn/while/lstm_cell/BiasAdd/EnterEnterq_eval/rnn/lstm_cell/bias/read*
parallel_iterations *
_output_shapes	
:*.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant(
~
 q_eval/rnn/while/lstm_cell/ConstConst^q_eval/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :

*q_eval/rnn/while/lstm_cell/split/split_dimConst^q_eval/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
љ
 q_eval/rnn/while/lstm_cell/splitSplit*q_eval/rnn/while/lstm_cell/split/split_dim"q_eval/rnn/while/lstm_cell/BiasAdd*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split

 q_eval/rnn/while/lstm_cell/add/yConst^q_eval/rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

q_eval/rnn/while/lstm_cell/addAdd"q_eval/rnn/while/lstm_cell/split:2 q_eval/rnn/while/lstm_cell/add/y*
T0*(
_output_shapes
:џџџџџџџџџ

"q_eval/rnn/while/lstm_cell/SigmoidSigmoidq_eval/rnn/while/lstm_cell/add*(
_output_shapes
:џџџџџџџџџ*
T0

q_eval/rnn/while/lstm_cell/mulMul"q_eval/rnn/while/lstm_cell/Sigmoidq_eval/rnn/while/Identity_3*(
_output_shapes
:џџџџџџџџџ*
T0

$q_eval/rnn/while/lstm_cell/Sigmoid_1Sigmoid q_eval/rnn/while/lstm_cell/split*(
_output_shapes
:џџџџџџџџџ*
T0
~
q_eval/rnn/while/lstm_cell/TanhTanh"q_eval/rnn/while/lstm_cell/split:1*(
_output_shapes
:џџџџџџџџџ*
T0
Ё
 q_eval/rnn/while/lstm_cell/mul_1Mul$q_eval/rnn/while/lstm_cell/Sigmoid_1q_eval/rnn/while/lstm_cell/Tanh*
T0*(
_output_shapes
:џџџџџџџџџ

 q_eval/rnn/while/lstm_cell/add_1Addq_eval/rnn/while/lstm_cell/mul q_eval/rnn/while/lstm_cell/mul_1*(
_output_shapes
:џџџџџџџџџ*
T0

$q_eval/rnn/while/lstm_cell/Sigmoid_2Sigmoid"q_eval/rnn/while/lstm_cell/split:3*(
_output_shapes
:џџџџџџџџџ*
T0
~
!q_eval/rnn/while/lstm_cell/Tanh_1Tanh q_eval/rnn/while/lstm_cell/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
Ѓ
 q_eval/rnn/while/lstm_cell/mul_2Mul$q_eval/rnn/while/lstm_cell/Sigmoid_2!q_eval/rnn/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:џџџџџџџџџ
щ
q_eval/rnn/while/SelectSelectq_eval/rnn/while/GreaterEqualq_eval/rnn/while/Select/Enter q_eval/rnn/while/lstm_cell/mul_2*(
_output_shapes
:џџџџџџџџџ*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2
ў
q_eval/rnn/while/Select/EnterEnterq_eval/rnn/zeros*
is_constant(*(
_output_shapes
:џџџџџџџџџ*.

frame_name q_eval/rnn/while/while_context*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
parallel_iterations 
щ
q_eval/rnn/while/Select_1Selectq_eval/rnn/while/GreaterEqualq_eval/rnn/while/Identity_3 q_eval/rnn/while/lstm_cell/add_1*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/add_1*(
_output_shapes
:џџџџџџџџџ
щ
q_eval/rnn/while/Select_2Selectq_eval/rnn/while/GreaterEqualq_eval/rnn/while/Identity_4 q_eval/rnn/while/lstm_cell/mul_2*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*(
_output_shapes
:џџџџџџџџџ
Џ
4q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3:q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterq_eval/rnn/while/Identity_1q_eval/rnn/while/Selectq_eval/rnn/while/Identity_2*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
_output_shapes
: 

:q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterq_eval/rnn/TensorArray*
is_constant(*
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
parallel_iterations 
v
q_eval/rnn/while/add_1/yConst^q_eval/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
q_eval/rnn/while/add_1Addq_eval/rnn/while/Identity_1q_eval/rnn/while/add_1/y*
_output_shapes
: *
T0
f
q_eval/rnn/while/NextIterationNextIterationq_eval/rnn/while/add*
T0*
_output_shapes
: 
j
 q_eval/rnn/while/NextIteration_1NextIterationq_eval/rnn/while/add_1*
T0*
_output_shapes
: 

 q_eval/rnn/while/NextIteration_2NextIteration4q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

 q_eval/rnn/while/NextIteration_3NextIterationq_eval/rnn/while/Select_1*
T0*(
_output_shapes
:џџџџџџџџџ

 q_eval/rnn/while/NextIteration_4NextIterationq_eval/rnn/while/Select_2*
T0*(
_output_shapes
:џџџџџџџџџ
W
q_eval/rnn/while/ExitExitq_eval/rnn/while/Switch*
T0*
_output_shapes
: 
[
q_eval/rnn/while/Exit_1Exitq_eval/rnn/while/Switch_1*
T0*
_output_shapes
: 
[
q_eval/rnn/while/Exit_2Exitq_eval/rnn/while/Switch_2*
_output_shapes
: *
T0
m
q_eval/rnn/while/Exit_3Exitq_eval/rnn/while/Switch_3*
T0*(
_output_shapes
:џџџџџџџџџ
m
q_eval/rnn/while/Exit_4Exitq_eval/rnn/while/Switch_4*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
-q_eval/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3q_eval/rnn/TensorArrayq_eval/rnn/while/Exit_2*)
_class
loc:@q_eval/rnn/TensorArray*
_output_shapes
: 

'q_eval/rnn/TensorArrayStack/range/startConst*)
_class
loc:@q_eval/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

'q_eval/rnn/TensorArrayStack/range/deltaConst*)
_class
loc:@q_eval/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 

!q_eval/rnn/TensorArrayStack/rangeRange'q_eval/rnn/TensorArrayStack/range/start-q_eval/rnn/TensorArrayStack/TensorArraySizeV3'q_eval/rnn/TensorArrayStack/range/delta*)
_class
loc:@q_eval/rnn/TensorArray*#
_output_shapes
:џџџџџџџџџ*

Tidx0
А
/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3q_eval/rnn/TensorArray!q_eval/rnn/TensorArrayStack/rangeq_eval/rnn/while/Exit_2*%
element_shape:џџџџџџџџџ*)
_class
loc:@q_eval/rnn/TensorArray*
dtype0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
]
q_eval/rnn/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
S
q_eval/rnn/Rank_3Const*
dtype0*
_output_shapes
: *
value	B :
Z
q_eval/rnn/range_3/startConst*
dtype0*
_output_shapes
: *
value	B :
Z
q_eval/rnn/range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_eval/rnn/range_3Rangeq_eval/rnn/range_3/startq_eval/rnn/Rank_3q_eval/rnn/range_3/delta*

Tidx0*
_output_shapes
:
m
q_eval/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Z
q_eval/rnn/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ѕ
q_eval/rnn/concat_2ConcatV2q_eval/rnn/concat_2/values_0q_eval/rnn/range_3q_eval/rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ж
q_eval/rnn/transpose_1	Transpose/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3q_eval/rnn/concat_2*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
Tperm0
Ѓ
/q_eval/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@q_eval/weights*
valueB"      

-q_eval/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *!
_class
loc:@q_eval/weights*
valueB
 *О

-q_eval/weights/Initializer/random_uniform/maxConst*!
_class
loc:@q_eval/weights*
valueB
 *>*
dtype0*
_output_shapes
: 
ь
7q_eval/weights/Initializer/random_uniform/RandomUniformRandomUniform/q_eval/weights/Initializer/random_uniform/shape*
T0*!
_class
loc:@q_eval/weights*
seed2 *
dtype0*
_output_shapes
:	*

seed 
ж
-q_eval/weights/Initializer/random_uniform/subSub-q_eval/weights/Initializer/random_uniform/max-q_eval/weights/Initializer/random_uniform/min*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
: 
щ
-q_eval/weights/Initializer/random_uniform/mulMul7q_eval/weights/Initializer/random_uniform/RandomUniform-q_eval/weights/Initializer/random_uniform/sub*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
:	
л
)q_eval/weights/Initializer/random_uniformAdd-q_eval/weights/Initializer/random_uniform/mul-q_eval/weights/Initializer/random_uniform/min*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
:	
Ї
q_eval/weights
VariableV2*!
_class
loc:@q_eval/weights*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
а
q_eval/weights/AssignAssignq_eval/weights)q_eval/weights/Initializer/random_uniform*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	*
use_locking(
|
q_eval/weights/readIdentityq_eval/weights*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
:	

/q_eval/weights/Regularizer/l2_regularizer/scaleConst*!
_class
loc:@q_eval/weights*
valueB
 *
з#<*
dtype0*
_output_shapes
: 

0q_eval/weights/Regularizer/l2_regularizer/L2LossL2Lossq_eval/weights/read*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
: 
з
)q_eval/weights/Regularizer/l2_regularizerMul/q_eval/weights/Regularizer/l2_regularizer/scale0q_eval/weights/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*!
_class
loc:@q_eval/weights

q_eval/biases/Initializer/ConstConst*
dtype0*
_output_shapes
:* 
_class
loc:@q_eval/biases*
valueB*
з#<

q_eval/biases
VariableV2*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@q_eval/biases*
	container *
shape:
О
q_eval/biases/AssignAssignq_eval/biasesq_eval/biases/Initializer/Const*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:
t
q_eval/biases/readIdentityq_eval/biases*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
:
o
q_eval/strided_slice/stackConst*!
valueB"    џџџџ    *
dtype0*
_output_shapes
:
q
q_eval/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*!
valueB"            
q
q_eval/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*!
valueB"         
И
q_eval/strided_sliceStridedSliceq_eval/rnn/transpose_1q_eval/strided_slice/stackq_eval/strided_slice/stack_1q_eval/strided_slice/stack_2*
end_mask*(
_output_shapes
:џџџџџџџџџ*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask 

q_eval/MatMulMatMulq_eval/strided_sliceq_eval/weights/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
f

q_eval/addAddq_eval/MatMulq_eval/biases/read*
T0*'
_output_shapes
:џџџџџџџџџ
a
q_eval/Q_value/tagConst*
valueB Bq_eval/Q_value*
dtype0*
_output_shapes
: 
c
q_eval/Q_valueHistogramSummaryq_eval/Q_value/tag
q_eval/add*
T0*
_output_shapes
: 
d

q_eval/MulMul
q_eval/addq_eval/action_taken*
T0*'
_output_shapes
:џџџџџџџџџ
^
q_eval/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 


q_eval/SumSum
q_eval/Mulq_eval/Sum/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
[

q_eval/subSubq_eval/q_value
q_eval/Sum*
T0*#
_output_shapes
:џџџџџџџџџ
Q
q_eval/SquareSquare
q_eval/sub*
T0*#
_output_shapes
:џџџџџџџџџ
V
q_eval/ConstConst*
valueB: *
dtype0*
_output_shapes
:
n
q_eval/MeanMeanq_eval/Squareq_eval/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
\
q_eval/Loss/tagsConst*
valueB Bq_eval/Loss*
dtype0*
_output_shapes
: 
\
q_eval/LossScalarSummaryq_eval/Loss/tagsq_eval/Mean*
_output_shapes
: *
T0
Y
q_eval/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
q_eval/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

q_eval/gradients/FillFillq_eval/gradients/Shapeq_eval/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
Z
q_eval/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
М
q_eval/gradients/f_count_1Enterq_eval/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context

q_eval/gradients/MergeMergeq_eval/gradients/f_count_1q_eval/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
w
q_eval/gradients/SwitchSwitchq_eval/gradients/Mergeq_eval/rnn/while/LoopCond*
_output_shapes
: : *
T0
t
q_eval/gradients/Add/yConst^q_eval/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
q_eval/gradients/AddAddq_eval/gradients/Switch:1q_eval/gradients/Add/y*
T0*
_output_shapes
: 
ъ
q_eval/gradients/NextIterationNextIterationq_eval/gradients/AddC^q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPushV2G^q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPushV2G^q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPushV2i^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2M^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2Y^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2[^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1W^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2K^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2Y^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2[^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1G^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2I^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2Y^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2[^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1G^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2I^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2W^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2Y^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1G^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
_output_shapes
: *
T0
\
q_eval/gradients/f_count_2Exitq_eval/gradients/Switch*
T0*
_output_shapes
: 
Z
q_eval/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
Я
q_eval/gradients/b_count_1Enterq_eval/gradients/f_count_2*
parallel_iterations *
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant( 

q_eval/gradients/Merge_1Mergeq_eval/gradients/b_count_1 q_eval/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 

q_eval/gradients/GreaterEqualGreaterEqualq_eval/gradients/Merge_1#q_eval/gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
ж
#q_eval/gradients/GreaterEqual/EnterEnterq_eval/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
]
q_eval/gradients/b_count_2LoopCondq_eval/gradients/GreaterEqual*
_output_shapes
: 
|
q_eval/gradients/Switch_1Switchq_eval/gradients/Merge_1q_eval/gradients/b_count_2*
T0*
_output_shapes
: : 
~
q_eval/gradients/SubSubq_eval/gradients/Switch_1:1#q_eval/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
Ю
 q_eval/gradients/NextIteration_1NextIterationq_eval/gradients/Subd^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
^
q_eval/gradients/b_count_3Exitq_eval/gradients/Switch_1*
T0*
_output_shapes
: 
y
/q_eval/gradients/q_eval/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Џ
)q_eval/gradients/q_eval/Mean_grad/ReshapeReshapeq_eval/gradients/Fill/q_eval/gradients/q_eval/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
t
'q_eval/gradients/q_eval/Mean_grad/ShapeShapeq_eval/Square*
_output_shapes
:*
T0*
out_type0
Т
&q_eval/gradients/q_eval/Mean_grad/TileTile)q_eval/gradients/q_eval/Mean_grad/Reshape'q_eval/gradients/q_eval/Mean_grad/Shape*
T0*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0
v
)q_eval/gradients/q_eval/Mean_grad/Shape_1Shapeq_eval/Square*
T0*
out_type0*
_output_shapes
:
l
)q_eval/gradients/q_eval/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
q
'q_eval/gradients/q_eval/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Р
&q_eval/gradients/q_eval/Mean_grad/ProdProd)q_eval/gradients/q_eval/Mean_grad/Shape_1'q_eval/gradients/q_eval/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
s
)q_eval/gradients/q_eval/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Ф
(q_eval/gradients/q_eval/Mean_grad/Prod_1Prod)q_eval/gradients/q_eval/Mean_grad/Shape_2)q_eval/gradients/q_eval/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
m
+q_eval/gradients/q_eval/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ќ
)q_eval/gradients/q_eval/Mean_grad/MaximumMaximum(q_eval/gradients/q_eval/Mean_grad/Prod_1+q_eval/gradients/q_eval/Mean_grad/Maximum/y*
_output_shapes
: *
T0
Њ
*q_eval/gradients/q_eval/Mean_grad/floordivFloorDiv&q_eval/gradients/q_eval/Mean_grad/Prod)q_eval/gradients/q_eval/Mean_grad/Maximum*
T0*
_output_shapes
: 

&q_eval/gradients/q_eval/Mean_grad/CastCast*q_eval/gradients/q_eval/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
В
)q_eval/gradients/q_eval/Mean_grad/truedivRealDiv&q_eval/gradients/q_eval/Mean_grad/Tile&q_eval/gradients/q_eval/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ

)q_eval/gradients/q_eval/Square_grad/ConstConst*^q_eval/gradients/q_eval/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 

'q_eval/gradients/q_eval/Square_grad/MulMul
q_eval/sub)q_eval/gradients/q_eval/Square_grad/Const*
T0*#
_output_shapes
:џџџџџџџџџ
В
)q_eval/gradients/q_eval/Square_grad/Mul_1Mul)q_eval/gradients/q_eval/Mean_grad/truediv'q_eval/gradients/q_eval/Square_grad/Mul*#
_output_shapes
:џџџџџџџџџ*
T0
t
&q_eval/gradients/q_eval/sub_grad/ShapeShapeq_eval/q_value*
T0*
out_type0*
_output_shapes
:
r
(q_eval/gradients/q_eval/sub_grad/Shape_1Shape
q_eval/Sum*
T0*
out_type0*
_output_shapes
:
о
6q_eval/gradients/q_eval/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&q_eval/gradients/q_eval/sub_grad/Shape(q_eval/gradients/q_eval/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
$q_eval/gradients/q_eval/sub_grad/SumSum)q_eval/gradients/q_eval/Square_grad/Mul_16q_eval/gradients/q_eval/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
(q_eval/gradients/q_eval/sub_grad/ReshapeReshape$q_eval/gradients/q_eval/sub_grad/Sum&q_eval/gradients/q_eval/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
в
&q_eval/gradients/q_eval/sub_grad/Sum_1Sum)q_eval/gradients/q_eval/Square_grad/Mul_18q_eval/gradients/q_eval/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
v
$q_eval/gradients/q_eval/sub_grad/NegNeg&q_eval/gradients/q_eval/sub_grad/Sum_1*
T0*
_output_shapes
:
С
*q_eval/gradients/q_eval/sub_grad/Reshape_1Reshape$q_eval/gradients/q_eval/sub_grad/Neg(q_eval/gradients/q_eval/sub_grad/Shape_1*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

1q_eval/gradients/q_eval/sub_grad/tuple/group_depsNoOp)^q_eval/gradients/q_eval/sub_grad/Reshape+^q_eval/gradients/q_eval/sub_grad/Reshape_1

9q_eval/gradients/q_eval/sub_grad/tuple/control_dependencyIdentity(q_eval/gradients/q_eval/sub_grad/Reshape2^q_eval/gradients/q_eval/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_eval/gradients/q_eval/sub_grad/Reshape*#
_output_shapes
:џџџџџџџџџ

;q_eval/gradients/q_eval/sub_grad/tuple/control_dependency_1Identity*q_eval/gradients/q_eval/sub_grad/Reshape_12^q_eval/gradients/q_eval/sub_grad/tuple/group_deps*#
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/sub_grad/Reshape_1
p
&q_eval/gradients/q_eval/Sum_grad/ShapeShape
q_eval/Mul*
T0*
out_type0*
_output_shapes
:
Ђ
%q_eval/gradients/q_eval/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B :
Ь
$q_eval/gradients/q_eval/Sum_grad/addAddq_eval/Sum/reduction_indices%q_eval/gradients/q_eval/Sum_grad/Size*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
_output_shapes
: 
й
$q_eval/gradients/q_eval/Sum_grad/modFloorMod$q_eval/gradients/q_eval/Sum_grad/add%q_eval/gradients/q_eval/Sum_grad/Size*
_output_shapes
: *
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape
І
(q_eval/gradients/q_eval/Sum_grad/Shape_1Const*
dtype0*
_output_shapes
: *9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
valueB 
Љ
,q_eval/gradients/q_eval/Sum_grad/range/startConst*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Љ
,q_eval/gradients/q_eval/Sum_grad/range/deltaConst*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

&q_eval/gradients/q_eval/Sum_grad/rangeRange,q_eval/gradients/q_eval/Sum_grad/range/start%q_eval/gradients/q_eval/Sum_grad/Size,q_eval/gradients/q_eval/Sum_grad/range/delta*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
_output_shapes
:*

Tidx0
Ј
+q_eval/gradients/q_eval/Sum_grad/Fill/valueConst*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ђ
%q_eval/gradients/q_eval/Sum_grad/FillFill(q_eval/gradients/q_eval/Sum_grad/Shape_1+q_eval/gradients/q_eval/Sum_grad/Fill/value*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*

index_type0*
_output_shapes
: 
Ю
.q_eval/gradients/q_eval/Sum_grad/DynamicStitchDynamicStitch&q_eval/gradients/q_eval/Sum_grad/range$q_eval/gradients/q_eval/Sum_grad/mod&q_eval/gradients/q_eval/Sum_grad/Shape%q_eval/gradients/q_eval/Sum_grad/Fill*
N*#
_output_shapes
:џџџџџџџџџ*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape
Ї
*q_eval/gradients/q_eval/Sum_grad/Maximum/yConst*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ј
(q_eval/gradients/q_eval/Sum_grad/MaximumMaximum.q_eval/gradients/q_eval/Sum_grad/DynamicStitch*q_eval/gradients/q_eval/Sum_grad/Maximum/y*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*#
_output_shapes
:џџџџџџџџџ
ч
)q_eval/gradients/q_eval/Sum_grad/floordivFloorDiv&q_eval/gradients/q_eval/Sum_grad/Shape(q_eval/gradients/q_eval/Sum_grad/Maximum*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
_output_shapes
:
б
(q_eval/gradients/q_eval/Sum_grad/ReshapeReshape;q_eval/gradients/q_eval/sub_grad/tuple/control_dependency_1.q_eval/gradients/q_eval/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
Ц
%q_eval/gradients/q_eval/Sum_grad/TileTile(q_eval/gradients/q_eval/Sum_grad/Reshape)q_eval/gradients/q_eval/Sum_grad/floordiv*
T0*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0
p
&q_eval/gradients/q_eval/Mul_grad/ShapeShape
q_eval/add*
_output_shapes
:*
T0*
out_type0
{
(q_eval/gradients/q_eval/Mul_grad/Shape_1Shapeq_eval/action_taken*
_output_shapes
:*
T0*
out_type0
о
6q_eval/gradients/q_eval/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs&q_eval/gradients/q_eval/Mul_grad/Shape(q_eval/gradients/q_eval/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

$q_eval/gradients/q_eval/Mul_grad/MulMul%q_eval/gradients/q_eval/Sum_grad/Tileq_eval/action_taken*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
$q_eval/gradients/q_eval/Mul_grad/SumSum$q_eval/gradients/q_eval/Mul_grad/Mul6q_eval/gradients/q_eval/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
С
(q_eval/gradients/q_eval/Mul_grad/ReshapeReshape$q_eval/gradients/q_eval/Mul_grad/Sum&q_eval/gradients/q_eval/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

&q_eval/gradients/q_eval/Mul_grad/Mul_1Mul
q_eval/add%q_eval/gradients/q_eval/Sum_grad/Tile*'
_output_shapes
:џџџџџџџџџ*
T0
Я
&q_eval/gradients/q_eval/Mul_grad/Sum_1Sum&q_eval/gradients/q_eval/Mul_grad/Mul_18q_eval/gradients/q_eval/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*q_eval/gradients/q_eval/Mul_grad/Reshape_1Reshape&q_eval/gradients/q_eval/Mul_grad/Sum_1(q_eval/gradients/q_eval/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

1q_eval/gradients/q_eval/Mul_grad/tuple/group_depsNoOp)^q_eval/gradients/q_eval/Mul_grad/Reshape+^q_eval/gradients/q_eval/Mul_grad/Reshape_1

9q_eval/gradients/q_eval/Mul_grad/tuple/control_dependencyIdentity(q_eval/gradients/q_eval/Mul_grad/Reshape2^q_eval/gradients/q_eval/Mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_eval/gradients/q_eval/Mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;q_eval/gradients/q_eval/Mul_grad/tuple/control_dependency_1Identity*q_eval/gradients/q_eval/Mul_grad/Reshape_12^q_eval/gradients/q_eval/Mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/Mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
s
&q_eval/gradients/q_eval/add_grad/ShapeShapeq_eval/MatMul*
T0*
out_type0*
_output_shapes
:
r
(q_eval/gradients/q_eval/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
о
6q_eval/gradients/q_eval/add_grad/BroadcastGradientArgsBroadcastGradientArgs&q_eval/gradients/q_eval/add_grad/Shape(q_eval/gradients/q_eval/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
о
$q_eval/gradients/q_eval/add_grad/SumSum9q_eval/gradients/q_eval/Mul_grad/tuple/control_dependency6q_eval/gradients/q_eval/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
С
(q_eval/gradients/q_eval/add_grad/ReshapeReshape$q_eval/gradients/q_eval/add_grad/Sum&q_eval/gradients/q_eval/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
т
&q_eval/gradients/q_eval/add_grad/Sum_1Sum9q_eval/gradients/q_eval/Mul_grad/tuple/control_dependency8q_eval/gradients/q_eval/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
К
*q_eval/gradients/q_eval/add_grad/Reshape_1Reshape&q_eval/gradients/q_eval/add_grad/Sum_1(q_eval/gradients/q_eval/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

1q_eval/gradients/q_eval/add_grad/tuple/group_depsNoOp)^q_eval/gradients/q_eval/add_grad/Reshape+^q_eval/gradients/q_eval/add_grad/Reshape_1

9q_eval/gradients/q_eval/add_grad/tuple/control_dependencyIdentity(q_eval/gradients/q_eval/add_grad/Reshape2^q_eval/gradients/q_eval/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*;
_class1
/-loc:@q_eval/gradients/q_eval/add_grad/Reshape

;q_eval/gradients/q_eval/add_grad/tuple/control_dependency_1Identity*q_eval/gradients/q_eval/add_grad/Reshape_12^q_eval/gradients/q_eval/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/add_grad/Reshape_1
н
*q_eval/gradients/q_eval/MatMul_grad/MatMulMatMul9q_eval/gradients/q_eval/add_grad/tuple/control_dependencyq_eval/weights/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
з
,q_eval/gradients/q_eval/MatMul_grad/MatMul_1MatMulq_eval/strided_slice9q_eval/gradients/q_eval/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(

4q_eval/gradients/q_eval/MatMul_grad/tuple/group_depsNoOp+^q_eval/gradients/q_eval/MatMul_grad/MatMul-^q_eval/gradients/q_eval/MatMul_grad/MatMul_1

<q_eval/gradients/q_eval/MatMul_grad/tuple/control_dependencyIdentity*q_eval/gradients/q_eval/MatMul_grad/MatMul5^q_eval/gradients/q_eval/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

>q_eval/gradients/q_eval/MatMul_grad/tuple/control_dependency_1Identity,q_eval/gradients/q_eval/MatMul_grad/MatMul_15^q_eval/gradients/q_eval/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@q_eval/gradients/q_eval/MatMul_grad/MatMul_1*
_output_shapes
:	

0q_eval/gradients/q_eval/strided_slice_grad/ShapeShapeq_eval/rnn/transpose_1*
T0*
out_type0*
_output_shapes
:
Ш
;q_eval/gradients/q_eval/strided_slice_grad/StridedSliceGradStridedSliceGrad0q_eval/gradients/q_eval/strided_slice_grad/Shapeq_eval/strided_slice/stackq_eval/strided_slice/stack_1q_eval/strided_slice/stack_2<q_eval/gradients/q_eval/MatMul_grad/tuple/control_dependency*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ

>q_eval/gradients/q_eval/rnn/transpose_1_grad/InvertPermutationInvertPermutationq_eval/rnn/concat_2*
T0*
_output_shapes
:

6q_eval/gradients/q_eval/rnn/transpose_1_grad/transpose	Transpose;q_eval/gradients/q_eval/strided_slice_grad/StridedSliceGrad>q_eval/gradients/q_eval/rnn/transpose_1_grad/InvertPermutation*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
Tperm0*
T0

gq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3q_eval/rnn/TensorArrayq_eval/rnn/while/Exit_2*)
_class
loc:@q_eval/rnn/TensorArray*
sourceq_eval/gradients*
_output_shapes

:: 
О
cq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityq_eval/rnn/while/Exit_2h^q_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*)
_class
loc:@q_eval/rnn/TensorArray*
_output_shapes
: 
Я
mq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3gq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3!q_eval/rnn/TensorArrayStack/range6q_eval/gradients/q_eval/rnn/transpose_1_grad/transposecq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
t
q_eval/gradients/zeros_like	ZerosLikeq_eval/rnn/while/Exit_3*
T0*(
_output_shapes
:џџџџџџџџџ
v
q_eval/gradients/zeros_like_1	ZerosLikeq_eval/rnn/while/Exit_4*
T0*(
_output_shapes
:џџџџџџџџџ
М
4q_eval/gradients/q_eval/rnn/while/Exit_2_grad/b_exitEntermq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ќ
4q_eval/gradients/q_eval/rnn/while/Exit_3_grad/b_exitEnterq_eval/gradients/zeros_like*
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant( 
ў
4q_eval/gradients/q_eval/rnn/while/Exit_4_grad/b_exitEnterq_eval/gradients/zeros_like_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ф
8q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switchMerge4q_eval/gradients/q_eval/rnn/while/Exit_2_grad/b_exit?q_eval/gradients/q_eval/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
і
8q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switchMerge4q_eval/gradients/q_eval/rnn/while/Exit_3_grad/b_exit?q_eval/gradients/q_eval/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
і
8q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switchMerge4q_eval/gradients/q_eval/rnn/while/Exit_4_grad/b_exit?q_eval/gradients/q_eval/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:џџџџџџџџџ: 

5q_eval/gradients/q_eval/rnn/while/Merge_2_grad/SwitchSwitch8q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switchq_eval/gradients/b_count_2*
_output_shapes
: : *
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switch

?q_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/group_depsNoOp6^q_eval/gradients/q_eval/rnn/while/Merge_2_grad/Switch
К
Gq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity5q_eval/gradients/q_eval/rnn/while/Merge_2_grad/Switch@^q_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
О
Iq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity7q_eval/gradients/q_eval/rnn/while/Merge_2_grad/Switch:1@^q_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
Љ
5q_eval/gradients/q_eval/rnn/while/Merge_3_grad/SwitchSwitch8q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switchq_eval/gradients/b_count_2*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switch*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

?q_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/group_depsNoOp6^q_eval/gradients/q_eval/rnn/while/Merge_3_grad/Switch
Ь
Gq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity5q_eval/gradients/q_eval/rnn/while/Merge_3_grad/Switch@^q_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
а
Iq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity7q_eval/gradients/q_eval/rnn/while/Merge_3_grad/Switch:1@^q_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
Љ
5q_eval/gradients/q_eval/rnn/while/Merge_4_grad/SwitchSwitch8q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switchq_eval/gradients/b_count_2*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switch*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

?q_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/group_depsNoOp6^q_eval/gradients/q_eval/rnn/while/Merge_4_grad/Switch
Ь
Gq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity5q_eval/gradients/q_eval/rnn/while/Merge_4_grad/Switch@^q_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switch
а
Iq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity7q_eval/gradients/q_eval/rnn/while/Merge_4_grad/Switch:1@^q_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switch
Ѕ
3q_eval/gradients/q_eval/rnn/while/Enter_2_grad/ExitExitGq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency*
_output_shapes
: *
T0
З
3q_eval/gradients/q_eval/rnn/while/Enter_3_grad/ExitExitGq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
З
3q_eval/gradients/q_eval/rnn/while/Enter_4_grad/ExitExitGq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Б
lq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterIq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
sourceq_eval/gradients*
_output_shapes

:: 
м
rq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterq_eval/rnn/TensorArray*
is_constant(*
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
parallel_iterations 

hq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityIq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1m^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2
щ
\q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3lq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3gq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2hq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:џџџџџџџџџ
н
bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@q_eval/rnn/while/Identity_1*
valueB :
џџџџџџџџџ
Р
bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*.
_class$
" loc:@q_eval/rnn/while/Identity_1*

stack_name *
_output_shapes
:
в
bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterbq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
У
hq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterq_eval/rnn/while/Identity_1^q_eval/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
Є
gq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2mq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^q_eval/gradients/Sub*
_output_shapes
: *
	elem_type0
ю
mq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterbq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
х
cq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerB^q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2F^q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2F^q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2h^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2L^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2X^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2Z^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1V^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2J^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2X^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2Z^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2H^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2X^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2Z^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2H^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2V^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2X^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2

[q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpJ^q_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1]^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Я
cq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentity\q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3\^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*o
_classe
caloc:@q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*(
_output_shapes
:џџџџџџџџџ

eq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityIq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1\^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
_output_shapes
: *
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switch
С
:q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like	ZerosLikeEq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
Л
@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@q_eval/rnn/while/Identity_3*
valueB :
џџџџџџџџџ
ќ
@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/f_accStackV2@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/Const*.
_class$
" loc:@q_eval/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0

@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/EnterEnter@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/Enterq_eval/rnn/while/Identity_3^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
Н
6q_eval/gradients/q_eval/rnn/while/Select_1_grad/SelectSelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2Iq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependency_1:q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like*(
_output_shapes
:џџџџџџџџџ*
T0
Й
<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/ConstConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@q_eval/rnn/while/GreaterEqual*
valueB :
џџџџџџџџџ
і
<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/f_accStackV2<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/Const*0
_class&
$"loc:@q_eval/rnn/while/GreaterEqual*

stack_name *
_output_shapes
:*
	elem_type0


<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/EnterEnter<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
ћ
Bq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPushV2StackPushV2<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/Enterq_eval/rnn/while/GreaterEqual^q_eval/gradients/Add*
T0
*
_output_shapes
:*
swap_memory( 
к
Aq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2
StackPopV2Gq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0
*
_output_shapes
:
Ђ
Gq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2/EnterEnter<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
П
8q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1SelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2:q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_likeIq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
М
@q_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/group_depsNoOp7^q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select9^q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1
Э
Hq_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/control_dependencyIdentity6q_eval/gradients/q_eval/rnn/while/Select_1_grad/SelectA^q_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select*(
_output_shapes
:џџџџџџџџџ
г
Jq_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/control_dependency_1Identity8q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1A^q_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1
С
:q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like	ZerosLikeEq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
Л
@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@q_eval/rnn/while/Identity_4*
valueB :
џџџџџџџџџ
ќ
@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/f_accStackV2@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/Const*.
_class$
" loc:@q_eval/rnn/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0

@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/EnterEnter@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/Enterq_eval/rnn/while/Identity_4^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
Н
6q_eval/gradients/q_eval/rnn/while/Select_2_grad/SelectSelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2Iq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependency_1:q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
П
8q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1SelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2:q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_likeIq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
М
@q_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/group_depsNoOp7^q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select9^q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1
Э
Hq_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/control_dependencyIdentity6q_eval/gradients/q_eval/rnn/while/Select_2_grad/SelectA^q_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select*(
_output_shapes
:џџџџџџџџџ
г
Jq_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/control_dependency_1Identity8q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1A^q_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1*(
_output_shapes
:џџџџџџџџџ
Я
8q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_like	ZerosLike>q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_like/Enter^q_eval/gradients/Sub*
T0*(
_output_shapes
:џџџџџџџџџ
ћ
>q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_like/EnterEnterq_eval/rnn/zeros*
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
г
4q_eval/gradients/q_eval/rnn/while/Select_grad/SelectSelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2cq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency8q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
е
6q_eval/gradients/q_eval/rnn/while/Select_grad/Select_1SelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV28q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_likecq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
>q_eval/gradients/q_eval/rnn/while/Select_grad/tuple/group_depsNoOp5^q_eval/gradients/q_eval/rnn/while/Select_grad/Select7^q_eval/gradients/q_eval/rnn/while/Select_grad/Select_1
Х
Fq_eval/gradients/q_eval/rnn/while/Select_grad/tuple/control_dependencyIdentity4q_eval/gradients/q_eval/rnn/while/Select_grad/Select?^q_eval/gradients/q_eval/rnn/while/Select_grad/tuple/group_deps*
T0*G
_class=
;9loc:@q_eval/gradients/q_eval/rnn/while/Select_grad/Select*(
_output_shapes
:џџџџџџџџџ
Ы
Hq_eval/gradients/q_eval/rnn/while/Select_grad/tuple/control_dependency_1Identity6q_eval/gradients/q_eval/rnn/while/Select_grad/Select_1?^q_eval/gradients/q_eval/rnn/while/Select_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_grad/Select_1

9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/ShapeShapeq_eval/rnn/zeros*
_output_shapes
:*
T0*
out_type0

?q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/zerosFill9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/Shape?q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ

9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_accEnter9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

;q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_acc_1Merge9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_accAq_eval/gradients/q_eval/rnn/while/Select/Enter_grad/NextIteration*
N**
_output_shapes
:џџџџџџџџџ: *
T0
ф
:q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/SwitchSwitch;q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_acc_1q_eval/gradients/b_count_2*
T0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
ї
7q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/AddAdd<q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/Switch:1Fq_eval/gradients/q_eval/rnn/while/Select_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
О
Aq_eval/gradients/q_eval/rnn/while/Select/Enter_grad/NextIterationNextIteration7q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/Add*
T0*(
_output_shapes
:џџџџџџџџџ
В
;q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_acc_2Exit:q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/Switch*
T0*(
_output_shapes
:џџџџџџџџџ
М
q_eval/gradients/AddNAddNJq_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/control_dependency_1Hq_eval/gradients/q_eval/rnn/while/Select_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1*
N*(
_output_shapes
:џџџџџџџџџ
 
<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/ShapeShape$q_eval/rnn/while/lstm_cell/Sigmoid_2*
_output_shapes
:*
T0*
out_type0

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape_1Shape!q_eval/rnn/while/lstm_cell/Tanh_1*
_output_shapes
:*
T0*
out_type0
ж
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsWq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ю
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape*
valueB :
џџџџџџџџџ
С
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
В
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ш
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
ђ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*
	elem_type0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:
Ж
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1EnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ю
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape_1^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
в
_q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
в
:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/MulMulq_eval/gradients/AddNEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
С
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/ConstConst*4
_class*
(&loc:@q_eval/rnn/while/lstm_cell/Tanh_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/f_accStackV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/Const*
	elem_type0*4
_class*
(&loc:@q_eval/rnn/while/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/Enter!q_eval/rnn/while/lstm_cell/Tanh_1^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/SumSum:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/MulLq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/ReshapeReshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/SumWq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ж
<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1MulGq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2q_eval/gradients/AddN*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*7
_class-
+)loc:@q_eval/rnn/while/lstm_cell/Sigmoid_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/Const*7
_class-
+)loc:@q_eval/rnn/while/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterBq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Hq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter$q_eval/rnn/while/lstm_cell/Sigmoid_2^q_eval/gradients/Add*(
_output_shapes
:џџџџџџџџџ*
swap_memory( *
T0
і
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ў
Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterBq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Sum_1Sum<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1Nq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѕ
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape_1Reshape<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Sum_1Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/ReshapeA^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape_1
ы
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentity>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/ReshapeH^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1Identity@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape_1H^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
и
?q_eval/gradients/q_eval/rnn/while/Switch_2_grad_1/NextIterationNextIterationeq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
}
,q_eval/gradients/q_eval/rnn/zeros_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
к
*q_eval/gradients/q_eval/rnn/zeros_grad/SumSum;q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_acc_2,q_eval/gradients/q_eval/rnn/zeros_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Ђ
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradGq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

@q_eval/gradients/q_eval/rnn/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
q_eval/gradients/AddN_1AddNJq_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/control_dependency_1@q_eval/gradients/q_eval/rnn/while/lstm_cell/Tanh_1_grad/TanhGrad*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1*
N*(
_output_shapes
:џџџџџџџџџ

<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/ShapeShapeq_eval/rnn/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape_1Shape q_eval/rnn/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
ж
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsWq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2Yq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ю
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape
В
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ш
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
Ю
]q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ђ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
Ж
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1EnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ю
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape_1^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
в
_q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ш
:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/SumSumq_eval/gradients/AddN_1Lq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/ReshapeReshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/SumWq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ь
<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Sum_1Sumq_eval/gradients/AddN_1Nq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape_1Reshape<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Sum_1Yq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/ReshapeA^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape_1
ы
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentity>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/ReshapeH^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape
ё
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1Identity@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape_1H^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/ShapeShape"q_eval/rnn/while/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape_1Shapeq_eval/rnn/while/Identity_3*
T0*
out_type0*
_output_shapes
:
а
Jq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsUq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*M
_classC
A?loc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*M
_classC
A?loc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape
Ў
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterPq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Т
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Uq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2[q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0*
_output_shapes
:
Ъ
[q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterPq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ю
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:
В
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1EnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ш
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape_1^q_eval/gradients/Add*
_output_shapes
:*
swap_memory( *
T0

Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
Ю
]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

8q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/MulMulOq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyEq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ

8q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/SumSum8q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/MulJq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/ReshapeReshape8q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/SumUq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1MulEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2Oq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
Т
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/ConstConst*5
_class+
)'loc:@q_eval/rnn/while/lstm_cell/Sigmoid*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/f_accStackV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*5
_class+
)'loc:@q_eval/rnn/while/lstm_cell/Sigmoid

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/Enter"q_eval/rnn/while/lstm_cell/Sigmoid^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Sum_1Sum:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1Lq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape_1Reshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Sum_1Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Э
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/group_depsNoOp=^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape?^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape_1
у
Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/control_dependencyIdentity<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/ReshapeF^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape
щ
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1Identity>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
 
<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/ShapeShape$q_eval/rnn/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape_1Shapeq_eval/rnn/while/lstm_cell/Tanh*
_output_shapes
:*
T0*
out_type0
ж
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsWq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ю
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape
В
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ш
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape^q_eval/gradients/Add*
_output_shapes
:*
swap_memory( *
T0

Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ђ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:
Ж
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1EnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ю
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape_1^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
в
_q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/MulMulQq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
П
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/ConstConst*2
_class(
&$loc:@q_eval/rnn/while/lstm_cell/Tanh*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/f_accStackV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/Const*2
_class(
&$loc:@q_eval/rnn/while/lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/Enterq_eval/rnn/while/lstm_cell/Tanh^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/SumSum:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/MulLq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/ReshapeReshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/SumWq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1MulGq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*7
_class-
+)loc:@q_eval/rnn/while/lstm_cell/Sigmoid_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*7
_class-
+)loc:@q_eval/rnn/while/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterBq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant(

Hq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter$q_eval/rnn/while/lstm_cell/Sigmoid_1^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
і
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ў
Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterBq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Sum_1Sum<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1Nq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape_1Reshape<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Sum_1Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/ReshapeA^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape_1
ы
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentity>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/ReshapeH^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1Identity@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape_1H^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape_1

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
С
q_eval/gradients/AddN_2AddNHq_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/control_dependencyOq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select
Ђ
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradGq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

>q_eval/gradients/q_eval/rnn/while/lstm_cell/Tanh_grad/TanhGradTanhGradEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ

:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/ShapeShape"q_eval/rnn/while/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:

<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape_1Const^q_eval/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
Е
Jq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsUq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ъ
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*M
_classC
A?loc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*M
_classC
A?loc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape
Ў
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterPq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant(
Т
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Uq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2[q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
Ъ
[q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterPq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(

8q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/SumSumDq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradJq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/ReshapeReshape8q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/SumUq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Sum_1SumDq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradLq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ђ
>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape_1Reshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Sum_1<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Э
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/group_depsNoOp=^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape?^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape_1
у
Mq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/control_dependencyIdentity<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/ReshapeF^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
з
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/control_dependency_1Identity>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 

?q_eval/gradients/q_eval/rnn/while/Switch_3_grad_1/NextIterationNextIterationq_eval/gradients/AddN_2*(
_output_shapes
:џџџџџџџџџ*
T0
ѕ
=q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concatConcatV2Fq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGrad>q_eval/gradients/q_eval/rnn/while/lstm_cell/Tanh_grad/TanhGradMq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/control_dependencyFq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradCq_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat/Const*

Tidx0*
T0*
N*(
_output_shapes
:џџџџџџџџџ

Cq_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat/ConstConst^q_eval/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
Я
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad=q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
и
Iq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpE^q_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad>^q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat
э
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity=q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concatJ^q_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat*(
_output_shapes
:џџџџџџџџџ
№
Sq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityDq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradJ^q_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
К
>q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMulMatMulQq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyDq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul/Enter*(
_output_shapes
:џџџџџџџџџ *
transpose_a( *
transpose_b(*
T0

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul/EnterEnter q_eval/rnn/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
 *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
Л
@q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1MatMulKq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
 *
transpose_a(*
transpose_b( *
T0
Ч
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*4
_class*
(&loc:@q_eval/rnn/while/lstm_cell/concat*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*4
_class*
(&loc:@q_eval/rnn/while/lstm_cell/concat*

stack_name *
_output_shapes
:

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterFq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ѓ
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter!q_eval/rnn/while/lstm_cell/concat^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ *
swap_memory( 
ў
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0*(
_output_shapes
:џџџџџџџџџ 
Ж
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterFq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
д
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMulA^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1
э
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentity>q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMulI^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ 
ы
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1Identity@q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1I^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
 *
T0*S
_classI
GEloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes	
:
Њ
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterDq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeFq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Lq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
р
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchFq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2q_eval/gradients/b_count_2*"
_output_shapes
::*
T0

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/AddAddGq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch:1Sq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ч
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationBq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:*
T0
Л
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitEq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:

=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ConstConst^q_eval/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

<q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/RankConst^q_eval/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
х
;q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/modFloorMod=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Const<q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 

=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeShape"q_eval/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeNShapeNIq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2Eq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2*
N* 
_output_shapes
::*
T0*
out_type0
Ц
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/ConstConst*5
_class+
)'loc:@q_eval/rnn/while/TensorArrayReadV3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/Const*

stack_name *
_output_shapes
:*
	elem_type0*5
_class+
)'loc:@q_eval/rnn/while/TensorArrayReadV3

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/EnterEnterDq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
 
Jq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/Enter"q_eval/rnn/while/TensorArrayReadV3^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ *
swap_memory( 
њ
Iq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Oq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ *
	elem_type0
В
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterDq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
О
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ConcatOffsetConcatOffset;q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/mod>q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN@q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
ц
=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/SliceSlicePq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyDq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ConcatOffset>q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ь
?q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice_1SlicePq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyFq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ConcatOffset:1@q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Index0*
T0
в
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/group_depsNoOp>^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice@^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice_1
ы
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyIdentity=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/SliceI^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice*(
_output_shapes
:џџџџџџџџџ 
ё
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1Identity?q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice_1I^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*R
_classH
FDloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice_1

Cq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_accConst*
dtype0* 
_output_shapes
:
 *
valueB
 *    
­
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterCq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
 *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

Eq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeEq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1Kq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
 : *
T0
ш
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchEq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2q_eval/gradients/b_count_2*
T0*,
_output_shapes
:
 :
 

Aq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/AddAddFq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/Switch:1Rq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
 
Ъ
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationAq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
 *
T0
О
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitDq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
 
Х
Zq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3`q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterbq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^q_eval/gradients/Sub*;
_class1
/-loc:@q_eval/rnn/while/TensorArrayReadV3/Enter*
sourceq_eval/gradients*
_output_shapes

:: 
д
`q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterq_eval/rnn/TensorArray_1*
T0*;
_class1
/-loc:@q_eval/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
џ
bq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterEq_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*;
_class1
/-loc:@q_eval/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

Vq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitybq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1[^q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*;
_class1
/-loc:@q_eval/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 

\q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Zq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3gq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyVq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
Ф
q_eval/gradients/AddN_3AddNHq_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/control_dependencyRq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select

Fq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
Hq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterFq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

Hq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeHq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Nq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
к
Gq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchHq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2q_eval/gradients/b_count_2*
T0*
_output_shapes
: : 

Dq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddIq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1\q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
Ц
Nq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationDq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
К
Hq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitGq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0

?q_eval/gradients/q_eval/rnn/while/Switch_4_grad_1/NextIterationNextIterationq_eval/gradients/AddN_3*
T0*(
_output_shapes
:џџџџџџџџџ
п
}q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3q_eval/rnn/TensorArray_1Hq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*+
_class!
loc:@q_eval/rnn/TensorArray_1*
sourceq_eval/gradients*
_output_shapes

:: 

yq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityHq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3~^q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*+
_class!
loc:@q_eval/rnn/TensorArray_1

oq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3}q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3#q_eval/rnn/TensorArrayUnstack/rangeyq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
element_shape:
Б
lq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpp^q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3I^q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
Ѕ
tq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityoq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3m^q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
T0*
_classx
vtloc:@q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
Й
vq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityHq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3m^q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 

<q_eval/gradients/q_eval/rnn/transpose_grad/InvertPermutationInvertPermutationq_eval/rnn/concat*
_output_shapes
:*
T0
Т
4q_eval/gradients/q_eval/rnn/transpose_grad/transpose	Transposetq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency<q_eval/gradients/q_eval/rnn/transpose_grad/InvertPermutation*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
Tperm0
z
,q_eval/gradients/q_eval/Reshape_1_grad/ShapeShapeq_eval/Reshape*
T0*
out_type0*
_output_shapes
:
о
.q_eval/gradients/q_eval/Reshape_1_grad/ReshapeReshape4q_eval/gradients/q_eval/rnn/transpose_grad/transpose,q_eval/gradients/q_eval/Reshape_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ 
w
*q_eval/gradients/q_eval/Reshape_grad/ShapeShapeq_eval/Relu_1*
_output_shapes
:*
T0*
out_type0
л
,q_eval/gradients/q_eval/Reshape_grad/ReshapeReshape.q_eval/gradients/q_eval/Reshape_1_grad/Reshape*q_eval/gradients/q_eval/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ		 
Џ
,q_eval/gradients/q_eval/Relu_1_grad/ReluGradReluGrad,q_eval/gradients/q_eval/Reshape_grad/Reshapeq_eval/Relu_1*
T0*/
_output_shapes
:џџџџџџџџџ		 
Џ
6q_eval/gradients/q_eval/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad,q_eval/gradients/q_eval/Relu_1_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
Ћ
;q_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/group_depsNoOp-^q_eval/gradients/q_eval/Relu_1_grad/ReluGrad7^q_eval/gradients/q_eval/conv2/BiasAdd_grad/BiasAddGrad
Ж
Cq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependencyIdentity,q_eval/gradients/q_eval/Relu_1_grad/ReluGrad<^q_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ		 *
T0*?
_class5
31loc:@q_eval/gradients/q_eval/Relu_1_grad/ReluGrad
З
Eq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependency_1Identity6q_eval/gradients/q_eval/conv2/BiasAdd_grad/BiasAddGrad<^q_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
Ѕ
0q_eval/gradients/q_eval/conv2/Conv2D_grad/ShapeNShapeNq_eval/Reluq_eval/conv2/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0

/q_eval/gradients/q_eval/conv2/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Љ
=q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput0q_eval/gradients/q_eval/conv2/Conv2D_grad/ShapeNq_eval/conv2/kernel/readCq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
љ
>q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterq_eval/Relu/q_eval/gradients/q_eval/conv2/Conv2D_grad/ConstCq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
: *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
У
:q_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropFilter>^q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropInput
ж
Bq_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/control_dependencyIdentity=q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropInput;^q_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ
б
Dq_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/control_dependency_1Identity>q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropFilter;^q_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
С
*q_eval/gradients/q_eval/Relu_grad/ReluGradReluGradBq_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/control_dependencyq_eval/Relu*/
_output_shapes
:џџџџџџџџџ*
T0
­
6q_eval/gradients/q_eval/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad*q_eval/gradients/q_eval/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
Љ
;q_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/group_depsNoOp+^q_eval/gradients/q_eval/Relu_grad/ReluGrad7^q_eval/gradients/q_eval/conv1/BiasAdd_grad/BiasAddGrad
В
Cq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependencyIdentity*q_eval/gradients/q_eval/Relu_grad/ReluGrad<^q_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/Relu_grad/ReluGrad
З
Eq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependency_1Identity6q_eval/gradients/q_eval/conv1/BiasAdd_grad/BiasAddGrad<^q_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ї
0q_eval/gradients/q_eval/conv1/Conv2D_grad/ShapeNShapeNq_eval/statesq_eval/conv1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::

/q_eval/gradients/q_eval/conv1/Conv2D_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"            
Љ
=q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput0q_eval/gradients/q_eval/conv1/Conv2D_grad/ShapeNq_eval/conv1/kernel/readCq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
ћ
>q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterq_eval/states/q_eval/gradients/q_eval/conv1/Conv2D_grad/ConstCq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
У
:q_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropFilter>^q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropInput
ж
Bq_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/control_dependencyIdentity=q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropInput;^q_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџTT
б
Dq_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/control_dependency_1Identity>q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropFilter;^q_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

 q_eval/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: * 
_class
loc:@q_eval/biases*
valueB
 *fff?

q_eval/beta1_power
VariableV2*
shared_name * 
_class
loc:@q_eval/biases*
	container *
shape: *
dtype0*
_output_shapes
: 
Х
q_eval/beta1_power/AssignAssignq_eval/beta1_power q_eval/beta1_power/initial_value*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: 
z
q_eval/beta1_power/readIdentityq_eval/beta1_power*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
: 

 q_eval/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: * 
_class
loc:@q_eval/biases*
valueB
 *wО?

q_eval/beta2_power
VariableV2*
shared_name * 
_class
loc:@q_eval/biases*
	container *
shape: *
dtype0*
_output_shapes
: 
Х
q_eval/beta2_power/AssignAssignq_eval/beta2_power q_eval/beta2_power/initial_value*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: *
use_locking(
z
q_eval/beta2_power/readIdentityq_eval/beta2_power*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
: 
Т
Aq_eval/q_eval/conv1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_eval/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Є
7q_eval/q_eval/conv1/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@q_eval/conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
 
1q_eval/q_eval/conv1/kernel/Adam/Initializer/zerosFillAq_eval/q_eval/conv1/kernel/Adam/Initializer/zeros/shape_as_tensor7q_eval/q_eval/conv1/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@q_eval/conv1/kernel*

index_type0*&
_output_shapes
:
Ы
q_eval/q_eval/conv1/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@q_eval/conv1/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:

&q_eval/q_eval/conv1/kernel/Adam/AssignAssignq_eval/q_eval/conv1/kernel/Adam1q_eval/q_eval/conv1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:
Њ
$q_eval/q_eval/conv1/kernel/Adam/readIdentityq_eval/q_eval/conv1/kernel/Adam*&
_output_shapes
:*
T0*&
_class
loc:@q_eval/conv1/kernel
Ф
Cq_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_eval/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
І
9q_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@q_eval/conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
І
3q_eval/q_eval/conv1/kernel/Adam_1/Initializer/zerosFillCq_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensor9q_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros/Const*&
_output_shapes
:*
T0*&
_class
loc:@q_eval/conv1/kernel*

index_type0
Э
!q_eval/q_eval/conv1/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@q_eval/conv1/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:

(q_eval/q_eval/conv1/kernel/Adam_1/AssignAssign!q_eval/q_eval/conv1/kernel/Adam_13q_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel
Ў
&q_eval/q_eval/conv1/kernel/Adam_1/readIdentity!q_eval/q_eval/conv1/kernel/Adam_1*
T0*&
_class
loc:@q_eval/conv1/kernel*&
_output_shapes
:
Ђ
/q_eval/q_eval/conv1/bias/Adam/Initializer/zerosConst*$
_class
loc:@q_eval/conv1/bias*
valueB*    *
dtype0*
_output_shapes
:
Џ
q_eval/q_eval/conv1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@q_eval/conv1/bias*
	container *
shape:
ђ
$q_eval/q_eval/conv1/bias/Adam/AssignAssignq_eval/q_eval/conv1/bias/Adam/q_eval/q_eval/conv1/bias/Adam/Initializer/zeros*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:*
use_locking(

"q_eval/q_eval/conv1/bias/Adam/readIdentityq_eval/q_eval/conv1/bias/Adam*
T0*$
_class
loc:@q_eval/conv1/bias*
_output_shapes
:
Є
1q_eval/q_eval/conv1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*$
_class
loc:@q_eval/conv1/bias*
valueB*    
Б
q_eval/q_eval/conv1/bias/Adam_1
VariableV2*
shared_name *$
_class
loc:@q_eval/conv1/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ј
&q_eval/q_eval/conv1/bias/Adam_1/AssignAssignq_eval/q_eval/conv1/bias/Adam_11q_eval/q_eval/conv1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:

$q_eval/q_eval/conv1/bias/Adam_1/readIdentityq_eval/q_eval/conv1/bias/Adam_1*
T0*$
_class
loc:@q_eval/conv1/bias*
_output_shapes
:
Т
Aq_eval/q_eval/conv2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_eval/conv2/kernel*%
valueB"             *
dtype0*
_output_shapes
:
Є
7q_eval/q_eval/conv2/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_eval/conv2/kernel*
valueB
 *    
 
1q_eval/q_eval/conv2/kernel/Adam/Initializer/zerosFillAq_eval/q_eval/conv2/kernel/Adam/Initializer/zeros/shape_as_tensor7q_eval/q_eval/conv2/kernel/Adam/Initializer/zeros/Const*&
_output_shapes
: *
T0*&
_class
loc:@q_eval/conv2/kernel*

index_type0
Ы
q_eval/q_eval/conv2/kernel/Adam
VariableV2*
shape: *
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_eval/conv2/kernel*
	container 

&q_eval/q_eval/conv2/kernel/Adam/AssignAssignq_eval/q_eval/conv2/kernel/Adam1q_eval/q_eval/conv2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: 
Њ
$q_eval/q_eval/conv2/kernel/Adam/readIdentityq_eval/q_eval/conv2/kernel/Adam*
T0*&
_class
loc:@q_eval/conv2/kernel*&
_output_shapes
: 
Ф
Cq_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_eval/conv2/kernel*%
valueB"             *
dtype0*
_output_shapes
:
І
9q_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@q_eval/conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
І
3q_eval/q_eval/conv2/kernel/Adam_1/Initializer/zerosFillCq_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensor9q_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@q_eval/conv2/kernel*

index_type0*&
_output_shapes
: 
Э
!q_eval/q_eval/conv2/kernel/Adam_1
VariableV2*
shape: *
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_eval/conv2/kernel*
	container 

(q_eval/q_eval/conv2/kernel/Adam_1/AssignAssign!q_eval/q_eval/conv2/kernel/Adam_13q_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(
Ў
&q_eval/q_eval/conv2/kernel/Adam_1/readIdentity!q_eval/q_eval/conv2/kernel/Adam_1*
T0*&
_class
loc:@q_eval/conv2/kernel*&
_output_shapes
: 
Ђ
/q_eval/q_eval/conv2/bias/Adam/Initializer/zerosConst*$
_class
loc:@q_eval/conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Џ
q_eval/q_eval/conv2/bias/Adam
VariableV2*
shared_name *$
_class
loc:@q_eval/conv2/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ђ
$q_eval/q_eval/conv2/bias/Adam/AssignAssignq_eval/q_eval/conv2/bias/Adam/q_eval/q_eval/conv2/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: 

"q_eval/q_eval/conv2/bias/Adam/readIdentityq_eval/q_eval/conv2/bias/Adam*
T0*$
_class
loc:@q_eval/conv2/bias*
_output_shapes
: 
Є
1q_eval/q_eval/conv2/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@q_eval/conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Б
q_eval/q_eval/conv2/bias/Adam_1
VariableV2*
shared_name *$
_class
loc:@q_eval/conv2/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ј
&q_eval/q_eval/conv2/bias/Adam_1/AssignAssignq_eval/q_eval/conv2/bias/Adam_11q_eval/q_eval/conv2/bias/Adam_1/Initializer/zeros*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: *
use_locking(

$q_eval/q_eval/conv2/bias/Adam_1/readIdentityq_eval/q_eval/conv2/bias/Adam_1*
T0*$
_class
loc:@q_eval/conv2/bias*
_output_shapes
: 
Ъ
Iq_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Д
?q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB
 *    
К
9q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zerosFillIq_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensor?q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
 *
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*

index_type0
Я
'q_eval/q_eval/rnn/lstm_cell/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
 *
shared_name *.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
	container *
shape:
 
 
.q_eval/q_eval/rnn/lstm_cell/kernel/Adam/AssignAssign'q_eval/q_eval/rnn/lstm_cell/kernel/Adam9q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
М
,q_eval/q_eval/rnn/lstm_cell/kernel/Adam/readIdentity'q_eval/q_eval/rnn/lstm_cell/kernel/Adam*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel* 
_output_shapes
:
 
Ь
Kq_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB"      
Ж
Aq_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB
 *    
Р
;q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zerosFillKq_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorAq_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*

index_type0* 
_output_shapes
:
 
б
)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
 *
shared_name *.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
	container *
shape:
 
І
0q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/AssignAssign)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1;q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
 *
use_locking(*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel
Р
.q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/readIdentity)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel* 
_output_shapes
:
 
Р
Gq_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
А
=q_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros/ConstConst*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
­
7q_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zerosFillGq_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensor=q_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*

index_type0
С
%q_eval/q_eval/rnn/lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
	container *
shape:

,q_eval/q_eval/rnn/lstm_cell/bias/Adam/AssignAssign%q_eval/q_eval/rnn/lstm_cell/bias/Adam7q_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias
Б
*q_eval/q_eval/rnn/lstm_cell/bias/Adam/readIdentity%q_eval/q_eval/rnn/lstm_cell/bias/Adam*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
_output_shapes	
:
Т
Iq_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
В
?q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/ConstConst*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
9q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zerosFillIq_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensor?q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
У
'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1
VariableV2*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

.q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/AssignAssign'q_eval/q_eval/rnn/lstm_cell/bias/Adam_19q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
Е
,q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/readIdentity'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
_output_shapes	
:
А
<q_eval/q_eval/weights/Adam/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@q_eval/weights*
valueB"      *
dtype0*
_output_shapes
:

2q_eval/q_eval/weights/Adam/Initializer/zeros/ConstConst*!
_class
loc:@q_eval/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

,q_eval/q_eval/weights/Adam/Initializer/zerosFill<q_eval/q_eval/weights/Adam/Initializer/zeros/shape_as_tensor2q_eval/q_eval/weights/Adam/Initializer/zeros/Const*
T0*!
_class
loc:@q_eval/weights*

index_type0*
_output_shapes
:	
Г
q_eval/q_eval/weights/Adam
VariableV2*
shared_name *!
_class
loc:@q_eval/weights*
	container *
shape:	*
dtype0*
_output_shapes
:	
ы
!q_eval/q_eval/weights/Adam/AssignAssignq_eval/q_eval/weights/Adam,q_eval/q_eval/weights/Adam/Initializer/zeros*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	*
use_locking(

q_eval/q_eval/weights/Adam/readIdentityq_eval/q_eval/weights/Adam*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
:	
В
>q_eval/q_eval/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@q_eval/weights*
valueB"      *
dtype0*
_output_shapes
:

4q_eval/q_eval/weights/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@q_eval/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

.q_eval/q_eval/weights/Adam_1/Initializer/zerosFill>q_eval/q_eval/weights/Adam_1/Initializer/zeros/shape_as_tensor4q_eval/q_eval/weights/Adam_1/Initializer/zeros/Const*
T0*!
_class
loc:@q_eval/weights*

index_type0*
_output_shapes
:	
Е
q_eval/q_eval/weights/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *!
_class
loc:@q_eval/weights*
	container *
shape:	
ё
#q_eval/q_eval/weights/Adam_1/AssignAssignq_eval/q_eval/weights/Adam_1.q_eval/q_eval/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	

!q_eval/q_eval/weights/Adam_1/readIdentityq_eval/q_eval/weights/Adam_1*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
:	

+q_eval/q_eval/biases/Adam/Initializer/zerosConst* 
_class
loc:@q_eval/biases*
valueB*    *
dtype0*
_output_shapes
:
Ї
q_eval/q_eval/biases/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@q_eval/biases
т
 q_eval/q_eval/biases/Adam/AssignAssignq_eval/q_eval/biases/Adam+q_eval/q_eval/biases/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:

q_eval/q_eval/biases/Adam/readIdentityq_eval/q_eval/biases/Adam*
_output_shapes
:*
T0* 
_class
loc:@q_eval/biases

-q_eval/q_eval/biases/Adam_1/Initializer/zerosConst* 
_class
loc:@q_eval/biases*
valueB*    *
dtype0*
_output_shapes
:
Љ
q_eval/q_eval/biases/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@q_eval/biases*
	container *
shape:
ш
"q_eval/q_eval/biases/Adam_1/AssignAssignq_eval/q_eval/biases/Adam_1-q_eval/q_eval/biases/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:

 q_eval/q_eval/biases/Adam_1/readIdentityq_eval/q_eval/biases/Adam_1*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
:
^
q_eval/Adam/learning_rateConst*
valueB
 *o9*
dtype0*
_output_shapes
: 
V
q_eval/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
V
q_eval/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
X
q_eval/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
ф
0q_eval/Adam/update_q_eval/conv1/kernel/ApplyAdam	ApplyAdamq_eval/conv1/kernelq_eval/q_eval/conv1/kernel/Adam!q_eval/q_eval/conv1/kernel/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonDq_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:*
use_locking( *
T0*&
_class
loc:@q_eval/conv1/kernel
Я
.q_eval/Adam/update_q_eval/conv1/bias/ApplyAdam	ApplyAdamq_eval/conv1/biasq_eval/q_eval/conv1/bias/Adamq_eval/q_eval/conv1/bias/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonEq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@q_eval/conv1/bias*
use_nesterov( *
_output_shapes
:
ф
0q_eval/Adam/update_q_eval/conv2/kernel/ApplyAdam	ApplyAdamq_eval/conv2/kernelq_eval/q_eval/conv2/kernel/Adam!q_eval/q_eval/conv2/kernel/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonDq_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@q_eval/conv2/kernel*
use_nesterov( *&
_output_shapes
: 
Я
.q_eval/Adam/update_q_eval/conv2/bias/ApplyAdam	ApplyAdamq_eval/conv2/biasq_eval/q_eval/conv2/bias/Adamq_eval/q_eval/conv2/bias/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonEq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@q_eval/conv2/bias*
use_nesterov( *
_output_shapes
: 

8q_eval/Adam/update_q_eval/rnn/lstm_cell/kernel/ApplyAdam	ApplyAdamq_eval/rnn/lstm_cell/kernel'q_eval/q_eval/rnn/lstm_cell/kernel/Adam)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonEq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
use_nesterov( * 
_output_shapes
:
 *
use_locking( *
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel
љ
6q_eval/Adam/update_q_eval/rnn/lstm_cell/bias/ApplyAdam	ApplyAdamq_eval/rnn/lstm_cell/bias%q_eval/q_eval/rnn/lstm_cell/bias/Adam'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonFq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
О
+q_eval/Adam/update_q_eval/weights/ApplyAdam	ApplyAdamq_eval/weightsq_eval/q_eval/weights/Adamq_eval/q_eval/weights/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilon>q_eval/gradients/q_eval/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	*
use_locking( *
T0*!
_class
loc:@q_eval/weights
Б
*q_eval/Adam/update_q_eval/biases/ApplyAdam	ApplyAdamq_eval/biasesq_eval/q_eval/biases/Adamq_eval/q_eval/biases/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilon;q_eval/gradients/q_eval/add_grad/tuple/control_dependency_1*
T0* 
_class
loc:@q_eval/biases*
use_nesterov( *
_output_shapes
:*
use_locking( 

q_eval/Adam/mulMulq_eval/beta1_power/readq_eval/Adam/beta1+^q_eval/Adam/update_q_eval/biases/ApplyAdam/^q_eval/Adam/update_q_eval/conv1/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv1/kernel/ApplyAdam/^q_eval/Adam/update_q_eval/conv2/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv2/kernel/ApplyAdam7^q_eval/Adam/update_q_eval/rnn/lstm_cell/bias/ApplyAdam9^q_eval/Adam/update_q_eval/rnn/lstm_cell/kernel/ApplyAdam,^q_eval/Adam/update_q_eval/weights/ApplyAdam*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
: 
­
q_eval/Adam/AssignAssignq_eval/beta1_powerq_eval/Adam/mul*
use_locking( *
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: 

q_eval/Adam/mul_1Mulq_eval/beta2_power/readq_eval/Adam/beta2+^q_eval/Adam/update_q_eval/biases/ApplyAdam/^q_eval/Adam/update_q_eval/conv1/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv1/kernel/ApplyAdam/^q_eval/Adam/update_q_eval/conv2/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv2/kernel/ApplyAdam7^q_eval/Adam/update_q_eval/rnn/lstm_cell/bias/ApplyAdam9^q_eval/Adam/update_q_eval/rnn/lstm_cell/kernel/ApplyAdam,^q_eval/Adam/update_q_eval/weights/ApplyAdam*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
: 
Б
q_eval/Adam/Assign_1Assignq_eval/beta2_powerq_eval/Adam/mul_1*
use_locking( *
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: 
ж
q_eval/AdamNoOp^q_eval/Adam/Assign^q_eval/Adam/Assign_1+^q_eval/Adam/update_q_eval/biases/ApplyAdam/^q_eval/Adam/update_q_eval/conv1/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv1/kernel/ApplyAdam/^q_eval/Adam/update_q_eval/conv2/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv2/kernel/ApplyAdam7^q_eval/Adam/update_q_eval/rnn/lstm_cell/bias/ApplyAdam9^q_eval/Adam/update_q_eval/rnn/lstm_cell/kernel/ApplyAdam,^q_eval/Adam/update_q_eval/weights/ApplyAdam
Ё
2q_eval/q_eval/conv1/kernel/summaries/histogram/tagConst*
dtype0*
_output_shapes
: *?
value6B4 B.q_eval/q_eval/conv1/kernel/summaries/histogram
Б
.q_eval/q_eval/conv1/kernel/summaries/histogramHistogramSummary2q_eval/q_eval/conv1/kernel/summaries/histogram/tagq_eval/conv1/kernel/read*
T0*
_output_shapes
: 

0q_eval/q_eval/conv1/bias/summaries/histogram/tagConst*=
value4B2 B,q_eval/q_eval/conv1/bias/summaries/histogram*
dtype0*
_output_shapes
: 
Ћ
,q_eval/q_eval/conv1/bias/summaries/histogramHistogramSummary0q_eval/q_eval/conv1/bias/summaries/histogram/tagq_eval/conv1/bias/read*
_output_shapes
: *
T0
Ё
2q_eval/q_eval/conv2/kernel/summaries/histogram/tagConst*?
value6B4 B.q_eval/q_eval/conv2/kernel/summaries/histogram*
dtype0*
_output_shapes
: 
Б
.q_eval/q_eval/conv2/kernel/summaries/histogramHistogramSummary2q_eval/q_eval/conv2/kernel/summaries/histogram/tagq_eval/conv2/kernel/read*
_output_shapes
: *
T0

0q_eval/q_eval/conv2/bias/summaries/histogram/tagConst*=
value4B2 B,q_eval/q_eval/conv2/bias/summaries/histogram*
dtype0*
_output_shapes
: 
Ћ
,q_eval/q_eval/conv2/bias/summaries/histogramHistogramSummary0q_eval/q_eval/conv2/bias/summaries/histogram/tagq_eval/conv2/bias/read*
T0*
_output_shapes
: 
Б
:q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram/tagConst*G
value>B< B6q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram*
dtype0*
_output_shapes
: 
Щ
6q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogramHistogramSummary:q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram/tag q_eval/rnn/lstm_cell/kernel/read*
T0*
_output_shapes
: 
­
8q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram/tagConst*E
value<B: B4q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram*
dtype0*
_output_shapes
: 
У
4q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogramHistogramSummary8q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram/tagq_eval/rnn/lstm_cell/bias/read*
T0*
_output_shapes
: 

-q_eval/q_eval/weights/summaries/histogram/tagConst*:
value1B/ B)q_eval/q_eval/weights/summaries/histogram*
dtype0*
_output_shapes
: 
Ђ
)q_eval/q_eval/weights/summaries/histogramHistogramSummary-q_eval/q_eval/weights/summaries/histogram/tagq_eval/weights/read*
T0*
_output_shapes
: 

,q_eval/q_eval/biases/summaries/histogram/tagConst*
dtype0*
_output_shapes
: *9
value0B. B(q_eval/q_eval/biases/summaries/histogram

(q_eval/q_eval/biases/summaries/histogramHistogramSummary,q_eval/q_eval/biases/summaries/histogram/tagq_eval/biases/read*
T0*
_output_shapes
: 
Щ
initNoOp^q_eval/beta1_power/Assign^q_eval/beta2_power/Assign^q_eval/biases/Assign^q_eval/conv1/bias/Assign^q_eval/conv1/kernel/Assign^q_eval/conv2/bias/Assign^q_eval/conv2/kernel/Assign!^q_eval/q_eval/biases/Adam/Assign#^q_eval/q_eval/biases/Adam_1/Assign%^q_eval/q_eval/conv1/bias/Adam/Assign'^q_eval/q_eval/conv1/bias/Adam_1/Assign'^q_eval/q_eval/conv1/kernel/Adam/Assign)^q_eval/q_eval/conv1/kernel/Adam_1/Assign%^q_eval/q_eval/conv2/bias/Adam/Assign'^q_eval/q_eval/conv2/bias/Adam_1/Assign'^q_eval/q_eval/conv2/kernel/Adam/Assign)^q_eval/q_eval/conv2/kernel/Adam_1/Assign-^q_eval/q_eval/rnn/lstm_cell/bias/Adam/Assign/^q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Assign/^q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Assign1^q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Assign"^q_eval/q_eval/weights/Adam/Assign$^q_eval/q_eval/weights/Adam_1/Assign!^q_eval/rnn/lstm_cell/bias/Assign#^q_eval/rnn/lstm_cell/kernel/Assign^q_eval/weights/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Я
save/SaveV2/tensor_namesConst*
valueјBѕBq_eval/beta1_powerBq_eval/beta2_powerBq_eval/biasesBq_eval/conv1/biasBq_eval/conv1/kernelBq_eval/conv2/biasBq_eval/conv2/kernelBq_eval/q_eval/biases/AdamBq_eval/q_eval/biases/Adam_1Bq_eval/q_eval/conv1/bias/AdamBq_eval/q_eval/conv1/bias/Adam_1Bq_eval/q_eval/conv1/kernel/AdamB!q_eval/q_eval/conv1/kernel/Adam_1Bq_eval/q_eval/conv2/bias/AdamBq_eval/q_eval/conv2/bias/Adam_1Bq_eval/q_eval/conv2/kernel/AdamB!q_eval/q_eval/conv2/kernel/Adam_1B%q_eval/q_eval/rnn/lstm_cell/bias/AdamB'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1B'q_eval/q_eval/rnn/lstm_cell/kernel/AdamB)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1Bq_eval/q_eval/weights/AdamBq_eval/q_eval/weights/Adam_1Bq_eval/rnn/lstm_cell/biasBq_eval/rnn/lstm_cell/kernelBq_eval/weights*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
№
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesq_eval/beta1_powerq_eval/beta2_powerq_eval/biasesq_eval/conv1/biasq_eval/conv1/kernelq_eval/conv2/biasq_eval/conv2/kernelq_eval/q_eval/biases/Adamq_eval/q_eval/biases/Adam_1q_eval/q_eval/conv1/bias/Adamq_eval/q_eval/conv1/bias/Adam_1q_eval/q_eval/conv1/kernel/Adam!q_eval/q_eval/conv1/kernel/Adam_1q_eval/q_eval/conv2/bias/Adamq_eval/q_eval/conv2/bias/Adam_1q_eval/q_eval/conv2/kernel/Adam!q_eval/q_eval/conv2/kernel/Adam_1%q_eval/q_eval/rnn/lstm_cell/bias/Adam'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1'q_eval/q_eval/rnn/lstm_cell/kernel/Adam)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1q_eval/q_eval/weights/Adamq_eval/q_eval/weights/Adam_1q_eval/rnn/lstm_cell/biasq_eval/rnn/lstm_cell/kernelq_eval/weights*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
с
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueјBѕBq_eval/beta1_powerBq_eval/beta2_powerBq_eval/biasesBq_eval/conv1/biasBq_eval/conv1/kernelBq_eval/conv2/biasBq_eval/conv2/kernelBq_eval/q_eval/biases/AdamBq_eval/q_eval/biases/Adam_1Bq_eval/q_eval/conv1/bias/AdamBq_eval/q_eval/conv1/bias/Adam_1Bq_eval/q_eval/conv1/kernel/AdamB!q_eval/q_eval/conv1/kernel/Adam_1Bq_eval/q_eval/conv2/bias/AdamBq_eval/q_eval/conv2/bias/Adam_1Bq_eval/q_eval/conv2/kernel/AdamB!q_eval/q_eval/conv2/kernel/Adam_1B%q_eval/q_eval/rnn/lstm_cell/bias/AdamB'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1B'q_eval/q_eval/rnn/lstm_cell/kernel/AdamB)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1Bq_eval/q_eval/weights/AdamBq_eval/q_eval/weights/Adam_1Bq_eval/rnn/lstm_cell/biasBq_eval/rnn/lstm_cell/kernelBq_eval/weights
Љ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
dtypes
2*|
_output_shapesj
h::::::::::::::::::::::::::
Ѕ
save/AssignAssignq_eval/beta1_powersave/RestoreV2*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: 
Љ
save/Assign_1Assignq_eval/beta2_powersave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@q_eval/biases
Ј
save/Assign_2Assignq_eval/biasessave/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:
А
save/Assign_3Assignq_eval/conv1/biassave/RestoreV2:3*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:
Р
save/Assign_4Assignq_eval/conv1/kernelsave/RestoreV2:4*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:
А
save/Assign_5Assignq_eval/conv2/biassave/RestoreV2:5*
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: 
Р
save/Assign_6Assignq_eval/conv2/kernelsave/RestoreV2:6*
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: 
Д
save/Assign_7Assignq_eval/q_eval/biases/Adamsave/RestoreV2:7*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:
Ж
save/Assign_8Assignq_eval/q_eval/biases/Adam_1save/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:
М
save/Assign_9Assignq_eval/q_eval/conv1/bias/Adamsave/RestoreV2:9*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:
Р
save/Assign_10Assignq_eval/q_eval/conv1/bias/Adam_1save/RestoreV2:10*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:
Ю
save/Assign_11Assignq_eval/q_eval/conv1/kernel/Adamsave/RestoreV2:11*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:
а
save/Assign_12Assign!q_eval/q_eval/conv1/kernel/Adam_1save/RestoreV2:12*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:
О
save/Assign_13Assignq_eval/q_eval/conv2/bias/Adamsave/RestoreV2:13*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias
Р
save/Assign_14Assignq_eval/q_eval/conv2/bias/Adam_1save/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: 
Ю
save/Assign_15Assignq_eval/q_eval/conv2/kernel/Adamsave/RestoreV2:15*
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: 
а
save/Assign_16Assign!q_eval/q_eval/conv2/kernel/Adam_1save/RestoreV2:16*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(
Я
save/Assign_17Assign%q_eval/q_eval/rnn/lstm_cell/bias/Adamsave/RestoreV2:17*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
б
save/Assign_18Assign'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1save/RestoreV2:18*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
и
save/Assign_19Assign'q_eval/q_eval/rnn/lstm_cell/kernel/Adamsave/RestoreV2:19*
validate_shape(* 
_output_shapes
:
 *
use_locking(*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel
к
save/Assign_20Assign)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1save/RestoreV2:20*
validate_shape(* 
_output_shapes
:
 *
use_locking(*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel
Н
save/Assign_21Assignq_eval/q_eval/weights/Adamsave/RestoreV2:21*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
П
save/Assign_22Assignq_eval/q_eval/weights/Adam_1save/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
У
save/Assign_23Assignq_eval/rnn/lstm_cell/biassave/RestoreV2:23*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
Ь
save/Assign_24Assignq_eval/rnn/lstm_cell/kernelsave/RestoreV2:24*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
Б
save/Assign_25Assignq_eval/weightssave/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
Ц
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
О
Merge/MergeSummaryMergeSummaryq_eval/Reward/Time_step_1#q_eval/TotalWaitingTime/Time_step_1q_eval/TotalDelay/Time_step_1q_eval/Q_valueq_eval/Loss.q_eval/q_eval/conv1/kernel/summaries/histogram,q_eval/q_eval/conv1/bias/summaries/histogram.q_eval/q_eval/conv2/kernel/summaries/histogram,q_eval/q_eval/conv2/bias/summaries/histogram6q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram4q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram)q_eval/q_eval/weights/summaries/histogram(q_eval/q_eval/biases/summaries/histogram*
N*
_output_shapes
: 

q_next/statesPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџTT*$
shape:џџџџџџџџџTT
v
q_next/action_takenPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
i
q_next/q_valuePlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
[
q_next/sequence_lengthPlaceholder*
shape:*
dtype0*
_output_shapes
:
V
q_next/batch_sizePlaceholder*
dtype0*
_output_shapes
:*
shape:
v
q_next/cell_statePlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
s
q_next/h_statePlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
X
q_next/Reward/Time_stepPlaceholder*
shape: *
dtype0*
_output_shapes
: 
x
q_next/Reward/Time_step_1/tagsConst*
dtype0*
_output_shapes
: **
value!B Bq_next/Reward/Time_step_1

q_next/Reward/Time_step_1ScalarSummaryq_next/Reward/Time_step_1/tagsq_next/Reward/Time_step*
T0*
_output_shapes
: 
b
!q_next/TotalWaitingTime/Time_stepPlaceholder*
dtype0*
_output_shapes
: *
shape: 

(q_next/TotalWaitingTime/Time_step_1/tagsConst*
dtype0*
_output_shapes
: *4
value+B) B#q_next/TotalWaitingTime/Time_step_1
Ђ
#q_next/TotalWaitingTime/Time_step_1ScalarSummary(q_next/TotalWaitingTime/Time_step_1/tags!q_next/TotalWaitingTime/Time_step*
T0*
_output_shapes
: 
\
q_next/TotalDelay/Time_stepPlaceholder*
shape: *
dtype0*
_output_shapes
: 

"q_next/TotalDelay/Time_step_1/tagsConst*.
value%B# Bq_next/TotalDelay/Time_step_1*
dtype0*
_output_shapes
: 

q_next/TotalDelay/Time_step_1ScalarSummary"q_next/TotalDelay/Time_step_1/tagsq_next/TotalDelay/Time_step*
T0*
_output_shapes
: 
З
6q_next/conv1/kernel/Initializer/truncated_normal/shapeConst*&
_class
loc:@q_next/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ђ
5q_next/conv1/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_next/conv1/kernel*
valueB
 *    
Є
7q_next/conv1/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_next/conv1/kernel*
valueB
 *аdN>

@q_next/conv1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6q_next/conv1/kernel/Initializer/truncated_normal/shape*

seed *
T0*&
_class
loc:@q_next/conv1/kernel*
seed2 *
dtype0*&
_output_shapes
:

4q_next/conv1/kernel/Initializer/truncated_normal/mulMul@q_next/conv1/kernel/Initializer/truncated_normal/TruncatedNormal7q_next/conv1/kernel/Initializer/truncated_normal/stddev*&
_output_shapes
:*
T0*&
_class
loc:@q_next/conv1/kernel
§
0q_next/conv1/kernel/Initializer/truncated_normalAdd4q_next/conv1/kernel/Initializer/truncated_normal/mul5q_next/conv1/kernel/Initializer/truncated_normal/mean*
T0*&
_class
loc:@q_next/conv1/kernel*&
_output_shapes
:
П
q_next/conv1/kernel
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *&
_class
loc:@q_next/conv1/kernel*
	container 
э
q_next/conv1/kernel/AssignAssignq_next/conv1/kernel0q_next/conv1/kernel/Initializer/truncated_normal*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_next/conv1/kernel

q_next/conv1/kernel/readIdentityq_next/conv1/kernel*
T0*&
_class
loc:@q_next/conv1/kernel*&
_output_shapes
:

#q_next/conv1/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*$
_class
loc:@q_next/conv1/bias*
valueB*
з#<
Ѓ
q_next/conv1/bias
VariableV2*
shared_name *$
_class
loc:@q_next/conv1/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Ю
q_next/conv1/bias/AssignAssignq_next/conv1/bias#q_next/conv1/bias/Initializer/Const*
T0*$
_class
loc:@q_next/conv1/bias*
validate_shape(*
_output_shapes
:*
use_locking(

q_next/conv1/bias/readIdentityq_next/conv1/bias*
T0*$
_class
loc:@q_next/conv1/bias*
_output_shapes
:
k
q_next/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
я
q_next/conv1/Conv2DConv2Dq_next/statesq_next/conv1/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ

q_next/conv1/BiasAddBiasAddq_next/conv1/Conv2Dq_next/conv1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ
c
q_next/ReluReluq_next/conv1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
З
6q_next/conv2/kernel/Initializer/truncated_normal/shapeConst*&
_class
loc:@q_next/conv2/kernel*%
valueB"             *
dtype0*
_output_shapes
:
Ђ
5q_next/conv2/kernel/Initializer/truncated_normal/meanConst*&
_class
loc:@q_next/conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
7q_next/conv2/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_next/conv2/kernel*
valueB
 *аdЮ=

@q_next/conv2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6q_next/conv2/kernel/Initializer/truncated_normal/shape*

seed *
T0*&
_class
loc:@q_next/conv2/kernel*
seed2 *
dtype0*&
_output_shapes
: 

4q_next/conv2/kernel/Initializer/truncated_normal/mulMul@q_next/conv2/kernel/Initializer/truncated_normal/TruncatedNormal7q_next/conv2/kernel/Initializer/truncated_normal/stddev*
T0*&
_class
loc:@q_next/conv2/kernel*&
_output_shapes
: 
§
0q_next/conv2/kernel/Initializer/truncated_normalAdd4q_next/conv2/kernel/Initializer/truncated_normal/mul5q_next/conv2/kernel/Initializer/truncated_normal/mean*
T0*&
_class
loc:@q_next/conv2/kernel*&
_output_shapes
: 
П
q_next/conv2/kernel
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_next/conv2/kernel*
	container *
shape: 
э
q_next/conv2/kernel/AssignAssignq_next/conv2/kernel0q_next/conv2/kernel/Initializer/truncated_normal*
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel*
validate_shape(*&
_output_shapes
: 

q_next/conv2/kernel/readIdentityq_next/conv2/kernel*
T0*&
_class
loc:@q_next/conv2/kernel*&
_output_shapes
: 

#q_next/conv2/bias/Initializer/ConstConst*$
_class
loc:@q_next/conv2/bias*
valueB *
з#<*
dtype0*
_output_shapes
: 
Ѓ
q_next/conv2/bias
VariableV2*$
_class
loc:@q_next/conv2/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Ю
q_next/conv2/bias/AssignAssignq_next/conv2/bias#q_next/conv2/bias/Initializer/Const*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias

q_next/conv2/bias/readIdentityq_next/conv2/bias*
T0*$
_class
loc:@q_next/conv2/bias*
_output_shapes
: 
k
q_next/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
э
q_next/conv2/Conv2DConv2Dq_next/Reluq_next/conv2/kernel/read*/
_output_shapes
:џџџџџџџџџ		 *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

q_next/conv2/BiasAddBiasAddq_next/conv2/Conv2Dq_next/conv2/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ		 *
T0
e
q_next/Relu_1Reluq_next/conv2/BiasAdd*/
_output_shapes
:џџџџџџџџџ		 *
T0
e
q_next/Reshape/shapeConst*
valueB"џџџџ 
  *
dtype0*
_output_shapes
:

q_next/ReshapeReshapeq_next/Relu_1q_next/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ 
[
q_next/Reshape_1/shape/2Const*
value
B : *
dtype0*
_output_shapes
: 

q_next/Reshape_1/shapePackq_next/batch_sizeq_next/sequence_lengthq_next/Reshape_1/shape/2*
T0*

axis *
N*
_output_shapes
:

q_next/Reshape_1Reshapeq_next/Reshapeq_next/Reshape_1/shape*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
T0*
Tshape0
Q
q_next/rnn/RankConst*
dtype0*
_output_shapes
: *
value	B :
X
q_next/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
X
q_next/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_next/rnn/rangeRangeq_next/rnn/range/startq_next/rnn/Rankq_next/rnn/range/delta*

Tidx0*
_output_shapes
:
k
q_next/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
X
q_next/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

q_next/rnn/concatConcatV2q_next/rnn/concat/values_0q_next/rnn/rangeq_next/rnn/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0

q_next/rnn/transpose	Transposeq_next/Reshape_1q_next/rnn/concat*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
Tperm0
a
q_next/rnn/sequence_lengthIdentityq_next/sequence_length*
T0*
_output_shapes
:
d
q_next/rnn/ShapeShapeq_next/rnn/transpose*
_output_shapes
:*
T0*
out_type0
h
q_next/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
j
 q_next/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
j
 q_next/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
А
q_next/rnn/strided_sliceStridedSliceq_next/rnn/Shapeq_next/rnn/strided_slice/stack q_next/rnn/strided_slice/stack_1 q_next/rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
u
q_next/rnn/Shape_1Shapeq_next/rnn/sequence_length*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
l
q_next/rnn/stackPackq_next/rnn/strided_slice*
N*
_output_shapes
:*
T0*

axis 
m
q_next/rnn/EqualEqualq_next/rnn/Shape_1q_next/rnn/stack*#
_output_shapes
:џџџџџџџџџ*
T0
Z
q_next/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
n
q_next/rnn/AllAllq_next/rnn/Equalq_next/rnn/Const*
_output_shapes
: *
	keep_dims( *

Tidx0

q_next/rnn/Assert/ConstConst*
dtype0*
_output_shapes
: *K
valueBB@ B:Expected shape for Tensor q_next/rnn/sequence_length:0 is 
j
q_next/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

q_next/rnn/Assert/Assert/data_0Const*K
valueBB@ B:Expected shape for Tensor q_next/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
p
q_next/rnn/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
К
q_next/rnn/Assert/AssertAssertq_next/rnn/Allq_next/rnn/Assert/Assert/data_0q_next/rnn/stackq_next/rnn/Assert/Assert/data_2q_next/rnn/Shape_1*
T
2*
	summarize
|
q_next/rnn/CheckSeqLenIdentityq_next/rnn/sequence_length^q_next/rnn/Assert/Assert*
T0*
_output_shapes
:
f
q_next/rnn/Shape_2Shapeq_next/rnn/transpose*
T0*
out_type0*
_output_shapes
:
j
 q_next/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
l
"q_next/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"q_next/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
К
q_next/rnn/strided_slice_1StridedSliceq_next/rnn/Shape_2 q_next/rnn/strided_slice_1/stack"q_next/rnn/strided_slice_1/stack_1"q_next/rnn/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
q_next/rnn/Shape_3Shapeq_next/rnn/transpose*
_output_shapes
:*
T0*
out_type0
j
 q_next/rnn/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
l
"q_next/rnn/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
l
"q_next/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
К
q_next/rnn/strided_slice_2StridedSliceq_next/rnn/Shape_3 q_next/rnn/strided_slice_2/stack"q_next/rnn/strided_slice_2/stack_1"q_next/rnn/strided_slice_2/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
[
q_next/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 

q_next/rnn/ExpandDims
ExpandDimsq_next/rnn/strided_slice_2q_next/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
]
q_next/rnn/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
Z
q_next/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

q_next/rnn/concat_1ConcatV2q_next/rnn/ExpandDimsq_next/rnn/Const_1q_next/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
[
q_next/rnn/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

q_next/rnn/zerosFillq_next/rnn/concat_1q_next/rnn/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
R
q_next/rnn/Rank_1Rankq_next/rnn/CheckSeqLen*
T0*
_output_shapes
: 
Z
q_next/rnn/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
Z
q_next/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_next/rnn/range_1Rangeq_next/rnn/range_1/startq_next/rnn/Rank_1q_next/rnn/range_1/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ

q_next/rnn/MinMinq_next/rnn/CheckSeqLenq_next/rnn/range_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
q_next/rnn/Rank_2Rankq_next/rnn/CheckSeqLen*
_output_shapes
: *
T0
Z
q_next/rnn/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
Z
q_next/rnn/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_next/rnn/range_2Rangeq_next/rnn/range_2/startq_next/rnn/Rank_2q_next/rnn/range_2/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0

q_next/rnn/MaxMaxq_next/rnn/CheckSeqLenq_next/rnn/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Q
q_next/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 

q_next/rnn/TensorArrayTensorArrayV3q_next/rnn/strided_slice_1*%
element_shape:џџџџџџџџџ*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*6
tensor_array_name!q_next/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 

q_next/rnn/TensorArray_1TensorArrayV3q_next/rnn/strided_slice_1*5
tensor_array_name q_next/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *%
element_shape:џџџџџџџџџ *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
w
#q_next/rnn/TensorArrayUnstack/ShapeShapeq_next/rnn/transpose*
T0*
out_type0*
_output_shapes
:
{
1q_next/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
}
3q_next/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
}
3q_next/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

+q_next/rnn/TensorArrayUnstack/strided_sliceStridedSlice#q_next/rnn/TensorArrayUnstack/Shape1q_next/rnn/TensorArrayUnstack/strided_slice/stack3q_next/rnn/TensorArrayUnstack/strided_slice/stack_13q_next/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
k
)q_next/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
k
)q_next/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
р
#q_next/rnn/TensorArrayUnstack/rangeRange)q_next/rnn/TensorArrayUnstack/range/start+q_next/rnn/TensorArrayUnstack/strided_slice)q_next/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0

Eq_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3q_next/rnn/TensorArray_1#q_next/rnn/TensorArrayUnstack/rangeq_next/rnn/transposeq_next/rnn/TensorArray_1:1*
T0*'
_class
loc:@q_next/rnn/transpose*
_output_shapes
: 
V
q_next/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
d
q_next/rnn/MaximumMaximumq_next/rnn/Maximum/xq_next/rnn/Max*
T0*
_output_shapes
: 
n
q_next/rnn/MinimumMinimumq_next/rnn/strided_slice_1q_next/rnn/Maximum*
T0*
_output_shapes
: 
d
"q_next/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Т
q_next/rnn/while/EnterEnter"q_next/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context
Б
q_next/rnn/while/Enter_1Enterq_next/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context
К
q_next/rnn/while/Enter_2Enterq_next/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context
Х
q_next/rnn/while/Enter_3Enterq_next/cell_state*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*.

frame_name q_next/rnn/while/while_context
Т
q_next/rnn/while/Enter_4Enterq_next/h_state*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*.

frame_name q_next/rnn/while/while_context

q_next/rnn/while/MergeMergeq_next/rnn/while/Enterq_next/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 

q_next/rnn/while/Merge_1Mergeq_next/rnn/while/Enter_1 q_next/rnn/while/NextIteration_1*
N*
_output_shapes
: : *
T0

q_next/rnn/while/Merge_2Mergeq_next/rnn/while/Enter_2 q_next/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

q_next/rnn/while/Merge_3Mergeq_next/rnn/while/Enter_3 q_next/rnn/while/NextIteration_3*
T0*
N**
_output_shapes
:џџџџџџџџџ: 

q_next/rnn/while/Merge_4Mergeq_next/rnn/while/Enter_4 q_next/rnn/while/NextIteration_4*
N**
_output_shapes
:џџџџџџџџџ: *
T0
s
q_next/rnn/while/LessLessq_next/rnn/while/Mergeq_next/rnn/while/Less/Enter*
T0*
_output_shapes
: 
П
q_next/rnn/while/Less/EnterEnterq_next/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context
y
q_next/rnn/while/Less_1Lessq_next/rnn/while/Merge_1q_next/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
Й
q_next/rnn/while/Less_1/EnterEnterq_next/rnn/Minimum*
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(
q
q_next/rnn/while/LogicalAnd
LogicalAndq_next/rnn/while/Lessq_next/rnn/while/Less_1*
_output_shapes
: 
Z
q_next/rnn/while/LoopCondLoopCondq_next/rnn/while/LogicalAnd*
_output_shapes
: 
Ђ
q_next/rnn/while/SwitchSwitchq_next/rnn/while/Mergeq_next/rnn/while/LoopCond*
T0*)
_class
loc:@q_next/rnn/while/Merge*
_output_shapes
: : 
Ј
q_next/rnn/while/Switch_1Switchq_next/rnn/while/Merge_1q_next/rnn/while/LoopCond*
T0*+
_class!
loc:@q_next/rnn/while/Merge_1*
_output_shapes
: : 
Ј
q_next/rnn/while/Switch_2Switchq_next/rnn/while/Merge_2q_next/rnn/while/LoopCond*
T0*+
_class!
loc:@q_next/rnn/while/Merge_2*
_output_shapes
: : 
Ь
q_next/rnn/while/Switch_3Switchq_next/rnn/while/Merge_3q_next/rnn/while/LoopCond*
T0*+
_class!
loc:@q_next/rnn/while/Merge_3*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Ь
q_next/rnn/while/Switch_4Switchq_next/rnn/while/Merge_4q_next/rnn/while/LoopCond*
T0*+
_class!
loc:@q_next/rnn/while/Merge_4*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
a
q_next/rnn/while/IdentityIdentityq_next/rnn/while/Switch:1*
T0*
_output_shapes
: 
e
q_next/rnn/while/Identity_1Identityq_next/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
e
q_next/rnn/while/Identity_2Identityq_next/rnn/while/Switch_2:1*
_output_shapes
: *
T0
w
q_next/rnn/while/Identity_3Identityq_next/rnn/while/Switch_3:1*
T0*(
_output_shapes
:џџџџџџџџџ
w
q_next/rnn/while/Identity_4Identityq_next/rnn/while/Switch_4:1*
T0*(
_output_shapes
:џџџџџџџџџ
t
q_next/rnn/while/add/yConst^q_next/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
q_next/rnn/while/addAddq_next/rnn/while/Identityq_next/rnn/while/add/y*
T0*
_output_shapes
: 
с
"q_next/rnn/while/TensorArrayReadV3TensorArrayReadV3(q_next/rnn/while/TensorArrayReadV3/Enterq_next/rnn/while/Identity_1*q_next/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:џџџџџџџџџ 
Ю
(q_next/rnn/while/TensorArrayReadV3/EnterEnterq_next/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
љ
*q_next/rnn/while/TensorArrayReadV3/Enter_1EnterEq_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context

q_next/rnn/while/GreaterEqualGreaterEqualq_next/rnn/while/Identity_1#q_next/rnn/while/GreaterEqual/Enter*
_output_shapes
:*
T0
Х
#q_next/rnn/while/GreaterEqual/EnterEnterq_next/rnn/CheckSeqLen*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(
Н
<q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB"      
Џ
:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB
 *њ<!Н
Џ
:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB
 *њ<!=

Dq_next/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform<q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
 *

seed *
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
seed2 

:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/subSub:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/max:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel

:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMulDq_next/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniform:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel* 
_output_shapes
:
 

6q_next/rnn/lstm_cell/kernel/Initializer/random_uniformAdd:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/mul:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel* 
_output_shapes
:
 
У
q_next/rnn/lstm_cell/kernel
VariableV2*
shape:
 *
dtype0* 
_output_shapes
:
 *
shared_name *.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
	container 

"q_next/rnn/lstm_cell/kernel/AssignAssignq_next/rnn/lstm_cell/kernel6q_next/rnn/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
t
 q_next/rnn/lstm_cell/kernel/readIdentityq_next/rnn/lstm_cell/kernel*
T0* 
_output_shapes
:
 
Д
;q_next/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB:
Є
1q_next/rnn/lstm_cell/bias/Initializer/zeros/ConstConst*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

+q_next/rnn/lstm_cell/bias/Initializer/zerosFill;q_next/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensor1q_next/rnn/lstm_cell/bias/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
Е
q_next/rnn/lstm_cell/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@q_next/rnn/lstm_cell/bias
я
 q_next/rnn/lstm_cell/bias/AssignAssignq_next/rnn/lstm_cell/bias+q_next/rnn/lstm_cell/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias
k
q_next/rnn/lstm_cell/bias/readIdentityq_next/rnn/lstm_cell/bias*
_output_shapes	
:*
T0

&q_next/rnn/while/lstm_cell/concat/axisConst^q_next/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
о
!q_next/rnn/while/lstm_cell/concatConcatV2"q_next/rnn/while/TensorArrayReadV3q_next/rnn/while/Identity_4&q_next/rnn/while/lstm_cell/concat/axis*
T0*
N*(
_output_shapes
:џџџџџџџџџ *

Tidx0
а
!q_next/rnn/while/lstm_cell/MatMulMatMul!q_next/rnn/while/lstm_cell/concat'q_next/rnn/while/lstm_cell/MatMul/Enter*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
л
'q_next/rnn/while/lstm_cell/MatMul/EnterEnter q_next/rnn/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
 *.

frame_name q_next/rnn/while/while_context
Ф
"q_next/rnn/while/lstm_cell/BiasAddBiasAdd!q_next/rnn/while/lstm_cell/MatMul(q_next/rnn/while/lstm_cell/BiasAdd/Enter*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
е
(q_next/rnn/while/lstm_cell/BiasAdd/EnterEnterq_next/rnn/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*.

frame_name q_next/rnn/while/while_context
~
 q_next/rnn/while/lstm_cell/ConstConst^q_next/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :

*q_next/rnn/while/lstm_cell/split/split_dimConst^q_next/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
љ
 q_next/rnn/while/lstm_cell/splitSplit*q_next/rnn/while/lstm_cell/split/split_dim"q_next/rnn/while/lstm_cell/BiasAdd*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split*
T0

 q_next/rnn/while/lstm_cell/add/yConst^q_next/rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

q_next/rnn/while/lstm_cell/addAdd"q_next/rnn/while/lstm_cell/split:2 q_next/rnn/while/lstm_cell/add/y*(
_output_shapes
:џџџџџџџџџ*
T0

"q_next/rnn/while/lstm_cell/SigmoidSigmoidq_next/rnn/while/lstm_cell/add*
T0*(
_output_shapes
:џџџџџџџџџ

q_next/rnn/while/lstm_cell/mulMul"q_next/rnn/while/lstm_cell/Sigmoidq_next/rnn/while/Identity_3*
T0*(
_output_shapes
:џџџџџџџџџ

$q_next/rnn/while/lstm_cell/Sigmoid_1Sigmoid q_next/rnn/while/lstm_cell/split*(
_output_shapes
:џџџџџџџџџ*
T0
~
q_next/rnn/while/lstm_cell/TanhTanh"q_next/rnn/while/lstm_cell/split:1*
T0*(
_output_shapes
:џџџџџџџџџ
Ё
 q_next/rnn/while/lstm_cell/mul_1Mul$q_next/rnn/while/lstm_cell/Sigmoid_1q_next/rnn/while/lstm_cell/Tanh*
T0*(
_output_shapes
:џџџџџџџџџ

 q_next/rnn/while/lstm_cell/add_1Addq_next/rnn/while/lstm_cell/mul q_next/rnn/while/lstm_cell/mul_1*
T0*(
_output_shapes
:џџџџџџџџџ

$q_next/rnn/while/lstm_cell/Sigmoid_2Sigmoid"q_next/rnn/while/lstm_cell/split:3*
T0*(
_output_shapes
:џџџџџџџџџ
~
!q_next/rnn/while/lstm_cell/Tanh_1Tanh q_next/rnn/while/lstm_cell/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ѓ
 q_next/rnn/while/lstm_cell/mul_2Mul$q_next/rnn/while/lstm_cell/Sigmoid_2!q_next/rnn/while/lstm_cell/Tanh_1*(
_output_shapes
:џџџџџџџџџ*
T0
щ
q_next/rnn/while/SelectSelectq_next/rnn/while/GreaterEqualq_next/rnn/while/Select/Enter q_next/rnn/while/lstm_cell/mul_2*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*(
_output_shapes
:џџџџџџџџџ
ў
q_next/rnn/while/Select/EnterEnterq_next/rnn/zeros*
is_constant(*(
_output_shapes
:џџџџџџџџџ*.

frame_name q_next/rnn/while/while_context*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*
parallel_iterations 
щ
q_next/rnn/while/Select_1Selectq_next/rnn/while/GreaterEqualq_next/rnn/while/Identity_3 q_next/rnn/while/lstm_cell/add_1*(
_output_shapes
:џџџџџџџџџ*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/add_1
щ
q_next/rnn/while/Select_2Selectq_next/rnn/while/GreaterEqualq_next/rnn/while/Identity_4 q_next/rnn/while/lstm_cell/mul_2*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*(
_output_shapes
:џџџџџџџџџ
Џ
4q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3:q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterq_next/rnn/while/Identity_1q_next/rnn/while/Selectq_next/rnn/while/Identity_2*
_output_shapes
: *
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2

:q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterq_next/rnn/TensorArray*
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(
v
q_next/rnn/while/add_1/yConst^q_next/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
q_next/rnn/while/add_1Addq_next/rnn/while/Identity_1q_next/rnn/while/add_1/y*
_output_shapes
: *
T0
f
q_next/rnn/while/NextIterationNextIterationq_next/rnn/while/add*
T0*
_output_shapes
: 
j
 q_next/rnn/while/NextIteration_1NextIterationq_next/rnn/while/add_1*
T0*
_output_shapes
: 

 q_next/rnn/while/NextIteration_2NextIteration4q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

 q_next/rnn/while/NextIteration_3NextIterationq_next/rnn/while/Select_1*(
_output_shapes
:џџџџџџџџџ*
T0

 q_next/rnn/while/NextIteration_4NextIterationq_next/rnn/while/Select_2*
T0*(
_output_shapes
:џџџџџџџџџ
W
q_next/rnn/while/ExitExitq_next/rnn/while/Switch*
T0*
_output_shapes
: 
[
q_next/rnn/while/Exit_1Exitq_next/rnn/while/Switch_1*
T0*
_output_shapes
: 
[
q_next/rnn/while/Exit_2Exitq_next/rnn/while/Switch_2*
T0*
_output_shapes
: 
m
q_next/rnn/while/Exit_3Exitq_next/rnn/while/Switch_3*
T0*(
_output_shapes
:џџџџџџџџџ
m
q_next/rnn/while/Exit_4Exitq_next/rnn/while/Switch_4*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
-q_next/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3q_next/rnn/TensorArrayq_next/rnn/while/Exit_2*)
_class
loc:@q_next/rnn/TensorArray*
_output_shapes
: 

'q_next/rnn/TensorArrayStack/range/startConst*)
_class
loc:@q_next/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

'q_next/rnn/TensorArrayStack/range/deltaConst*)
_class
loc:@q_next/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 

!q_next/rnn/TensorArrayStack/rangeRange'q_next/rnn/TensorArrayStack/range/start-q_next/rnn/TensorArrayStack/TensorArraySizeV3'q_next/rnn/TensorArrayStack/range/delta*)
_class
loc:@q_next/rnn/TensorArray*#
_output_shapes
:џџџџџџџџџ*

Tidx0
А
/q_next/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3q_next/rnn/TensorArray!q_next/rnn/TensorArrayStack/rangeq_next/rnn/while/Exit_2*
dtype0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*%
element_shape:џџџџџџџџџ*)
_class
loc:@q_next/rnn/TensorArray
]
q_next/rnn/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
S
q_next/rnn/Rank_3Const*
value	B :*
dtype0*
_output_shapes
: 
Z
q_next/rnn/range_3/startConst*
dtype0*
_output_shapes
: *
value	B :
Z
q_next/rnn/range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_next/rnn/range_3Rangeq_next/rnn/range_3/startq_next/rnn/Rank_3q_next/rnn/range_3/delta*
_output_shapes
:*

Tidx0
m
q_next/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Z
q_next/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
q_next/rnn/concat_2ConcatV2q_next/rnn/concat_2/values_0q_next/rnn/range_3q_next/rnn/concat_2/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ж
q_next/rnn/transpose_1	Transpose/q_next/rnn/TensorArrayStack/TensorArrayGatherV3q_next/rnn/concat_2*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
Tperm0
Ѓ
/q_next/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@q_next/weights*
valueB"      

-q_next/weights/Initializer/random_uniform/minConst*!
_class
loc:@q_next/weights*
valueB
 *О*
dtype0*
_output_shapes
: 

-q_next/weights/Initializer/random_uniform/maxConst*!
_class
loc:@q_next/weights*
valueB
 *>*
dtype0*
_output_shapes
: 
ь
7q_next/weights/Initializer/random_uniform/RandomUniformRandomUniform/q_next/weights/Initializer/random_uniform/shape*
T0*!
_class
loc:@q_next/weights*
seed2 *
dtype0*
_output_shapes
:	*

seed 
ж
-q_next/weights/Initializer/random_uniform/subSub-q_next/weights/Initializer/random_uniform/max-q_next/weights/Initializer/random_uniform/min*
T0*!
_class
loc:@q_next/weights*
_output_shapes
: 
щ
-q_next/weights/Initializer/random_uniform/mulMul7q_next/weights/Initializer/random_uniform/RandomUniform-q_next/weights/Initializer/random_uniform/sub*
T0*!
_class
loc:@q_next/weights*
_output_shapes
:	
л
)q_next/weights/Initializer/random_uniformAdd-q_next/weights/Initializer/random_uniform/mul-q_next/weights/Initializer/random_uniform/min*
T0*!
_class
loc:@q_next/weights*
_output_shapes
:	
Ї
q_next/weights
VariableV2*
shared_name *!
_class
loc:@q_next/weights*
	container *
shape:	*
dtype0*
_output_shapes
:	
а
q_next/weights/AssignAssignq_next/weights)q_next/weights/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	
|
q_next/weights/readIdentityq_next/weights*
T0*!
_class
loc:@q_next/weights*
_output_shapes
:	

/q_next/weights/Regularizer/l2_regularizer/scaleConst*!
_class
loc:@q_next/weights*
valueB
 *
з#<*
dtype0*
_output_shapes
: 

0q_next/weights/Regularizer/l2_regularizer/L2LossL2Lossq_next/weights/read*
T0*!
_class
loc:@q_next/weights*
_output_shapes
: 
з
)q_next/weights/Regularizer/l2_regularizerMul/q_next/weights/Regularizer/l2_regularizer/scale0q_next/weights/Regularizer/l2_regularizer/L2Loss*
T0*!
_class
loc:@q_next/weights*
_output_shapes
: 

q_next/biases/Initializer/ConstConst* 
_class
loc:@q_next/biases*
valueB*
з#<*
dtype0*
_output_shapes
:

q_next/biases
VariableV2*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@q_next/biases*
	container *
shape:
О
q_next/biases/AssignAssignq_next/biasesq_next/biases/Initializer/Const*
use_locking(*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
:
t
q_next/biases/readIdentityq_next/biases*
T0* 
_class
loc:@q_next/biases*
_output_shapes
:
o
q_next/strided_slice/stackConst*!
valueB"    џџџџ    *
dtype0*
_output_shapes
:
q
q_next/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*!
valueB"            
q
q_next/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
И
q_next/strided_sliceStridedSliceq_next/rnn/transpose_1q_next/strided_slice/stackq_next/strided_slice/stack_1q_next/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*(
_output_shapes
:џџџџџџџџџ*
Index0*
T0

q_next/MatMulMatMulq_next/strided_sliceq_next/weights/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
f

q_next/addAddq_next/MatMulq_next/biases/read*'
_output_shapes
:џџџџџџџџџ*
T0
a
q_next/Q_value/tagConst*
valueB Bq_next/Q_value*
dtype0*
_output_shapes
: 
c
q_next/Q_valueHistogramSummaryq_next/Q_value/tag
q_next/add*
T0*
_output_shapes
: 
d

q_next/MulMul
q_next/addq_next/action_taken*'
_output_shapes
:џџџџџџџџџ*
T0
^
q_next/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 


q_next/SumSum
q_next/Mulq_next/Sum/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
[

q_next/subSubq_next/q_value
q_next/Sum*
T0*#
_output_shapes
:џџџџџџџџџ
Q
q_next/SquareSquare
q_next/sub*#
_output_shapes
:џџџџџџџџџ*
T0
V
q_next/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
n
q_next/MeanMeanq_next/Squareq_next/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
\
q_next/Loss/tagsConst*
valueB Bq_next/Loss*
dtype0*
_output_shapes
: 
\
q_next/LossScalarSummaryq_next/Loss/tagsq_next/Mean*
_output_shapes
: *
T0
Y
q_next/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
_
q_next/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

q_next/gradients/FillFillq_next/gradients/Shapeq_next/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Z
q_next/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
М
q_next/gradients/f_count_1Enterq_next/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context

q_next/gradients/MergeMergeq_next/gradients/f_count_1q_next/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
w
q_next/gradients/SwitchSwitchq_next/gradients/Mergeq_next/rnn/while/LoopCond*
_output_shapes
: : *
T0
t
q_next/gradients/Add/yConst^q_next/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
o
q_next/gradients/AddAddq_next/gradients/Switch:1q_next/gradients/Add/y*
T0*
_output_shapes
: 
ъ
q_next/gradients/NextIterationNextIterationq_next/gradients/AddC^q_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPushV2G^q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPushV2G^q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPushV2i^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2M^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2Y^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2[^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1W^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2K^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2Y^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2[^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1G^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2I^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2Y^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2[^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1G^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2I^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2W^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2Y^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1G^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
\
q_next/gradients/f_count_2Exitq_next/gradients/Switch*
T0*
_output_shapes
: 
Z
q_next/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
Я
q_next/gradients/b_count_1Enterq_next/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

q_next/gradients/Merge_1Mergeq_next/gradients/b_count_1 q_next/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 

q_next/gradients/GreaterEqualGreaterEqualq_next/gradients/Merge_1#q_next/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
ж
#q_next/gradients/GreaterEqual/EnterEnterq_next/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
]
q_next/gradients/b_count_2LoopCondq_next/gradients/GreaterEqual*
_output_shapes
: 
|
q_next/gradients/Switch_1Switchq_next/gradients/Merge_1q_next/gradients/b_count_2*
T0*
_output_shapes
: : 
~
q_next/gradients/SubSubq_next/gradients/Switch_1:1#q_next/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
Ю
 q_next/gradients/NextIteration_1NextIterationq_next/gradients/Subd^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
^
q_next/gradients/b_count_3Exitq_next/gradients/Switch_1*
_output_shapes
: *
T0
y
/q_next/gradients/q_next/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Џ
)q_next/gradients/q_next/Mean_grad/ReshapeReshapeq_next/gradients/Fill/q_next/gradients/q_next/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
t
'q_next/gradients/q_next/Mean_grad/ShapeShapeq_next/Square*
_output_shapes
:*
T0*
out_type0
Т
&q_next/gradients/q_next/Mean_grad/TileTile)q_next/gradients/q_next/Mean_grad/Reshape'q_next/gradients/q_next/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
v
)q_next/gradients/q_next/Mean_grad/Shape_1Shapeq_next/Square*
_output_shapes
:*
T0*
out_type0
l
)q_next/gradients/q_next/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
q
'q_next/gradients/q_next/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Р
&q_next/gradients/q_next/Mean_grad/ProdProd)q_next/gradients/q_next/Mean_grad/Shape_1'q_next/gradients/q_next/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
s
)q_next/gradients/q_next/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ф
(q_next/gradients/q_next/Mean_grad/Prod_1Prod)q_next/gradients/q_next/Mean_grad/Shape_2)q_next/gradients/q_next/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
m
+q_next/gradients/q_next/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
Ќ
)q_next/gradients/q_next/Mean_grad/MaximumMaximum(q_next/gradients/q_next/Mean_grad/Prod_1+q_next/gradients/q_next/Mean_grad/Maximum/y*
_output_shapes
: *
T0
Њ
*q_next/gradients/q_next/Mean_grad/floordivFloorDiv&q_next/gradients/q_next/Mean_grad/Prod)q_next/gradients/q_next/Mean_grad/Maximum*
T0*
_output_shapes
: 

&q_next/gradients/q_next/Mean_grad/CastCast*q_next/gradients/q_next/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
В
)q_next/gradients/q_next/Mean_grad/truedivRealDiv&q_next/gradients/q_next/Mean_grad/Tile&q_next/gradients/q_next/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ

)q_next/gradients/q_next/Square_grad/ConstConst*^q_next/gradients/q_next/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 

'q_next/gradients/q_next/Square_grad/MulMul
q_next/sub)q_next/gradients/q_next/Square_grad/Const*
T0*#
_output_shapes
:џџџџџџџџџ
В
)q_next/gradients/q_next/Square_grad/Mul_1Mul)q_next/gradients/q_next/Mean_grad/truediv'q_next/gradients/q_next/Square_grad/Mul*
T0*#
_output_shapes
:џџџџџџџџџ
t
&q_next/gradients/q_next/sub_grad/ShapeShapeq_next/q_value*
T0*
out_type0*
_output_shapes
:
r
(q_next/gradients/q_next/sub_grad/Shape_1Shape
q_next/Sum*
T0*
out_type0*
_output_shapes
:
о
6q_next/gradients/q_next/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&q_next/gradients/q_next/sub_grad/Shape(q_next/gradients/q_next/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
$q_next/gradients/q_next/sub_grad/SumSum)q_next/gradients/q_next/Square_grad/Mul_16q_next/gradients/q_next/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
(q_next/gradients/q_next/sub_grad/ReshapeReshape$q_next/gradients/q_next/sub_grad/Sum&q_next/gradients/q_next/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
в
&q_next/gradients/q_next/sub_grad/Sum_1Sum)q_next/gradients/q_next/Square_grad/Mul_18q_next/gradients/q_next/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
v
$q_next/gradients/q_next/sub_grad/NegNeg&q_next/gradients/q_next/sub_grad/Sum_1*
_output_shapes
:*
T0
С
*q_next/gradients/q_next/sub_grad/Reshape_1Reshape$q_next/gradients/q_next/sub_grad/Neg(q_next/gradients/q_next/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

1q_next/gradients/q_next/sub_grad/tuple/group_depsNoOp)^q_next/gradients/q_next/sub_grad/Reshape+^q_next/gradients/q_next/sub_grad/Reshape_1

9q_next/gradients/q_next/sub_grad/tuple/control_dependencyIdentity(q_next/gradients/q_next/sub_grad/Reshape2^q_next/gradients/q_next/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_next/gradients/q_next/sub_grad/Reshape*#
_output_shapes
:џџџџџџџџџ

;q_next/gradients/q_next/sub_grad/tuple/control_dependency_1Identity*q_next/gradients/q_next/sub_grad/Reshape_12^q_next/gradients/q_next/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_next/gradients/q_next/sub_grad/Reshape_1*#
_output_shapes
:џџџџџџџџџ
p
&q_next/gradients/q_next/Sum_grad/ShapeShape
q_next/Mul*
T0*
out_type0*
_output_shapes
:
Ђ
%q_next/gradients/q_next/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B :
Ь
$q_next/gradients/q_next/Sum_grad/addAddq_next/Sum/reduction_indices%q_next/gradients/q_next/Sum_grad/Size*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
_output_shapes
: 
й
$q_next/gradients/q_next/Sum_grad/modFloorMod$q_next/gradients/q_next/Sum_grad/add%q_next/gradients/q_next/Sum_grad/Size*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
_output_shapes
: 
І
(q_next/gradients/q_next/Sum_grad/Shape_1Const*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
Љ
,q_next/gradients/q_next/Sum_grad/range/startConst*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Љ
,q_next/gradients/q_next/Sum_grad/range/deltaConst*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

&q_next/gradients/q_next/Sum_grad/rangeRange,q_next/gradients/q_next/Sum_grad/range/start%q_next/gradients/q_next/Sum_grad/Size,q_next/gradients/q_next/Sum_grad/range/delta*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
_output_shapes
:*

Tidx0
Ј
+q_next/gradients/q_next/Sum_grad/Fill/valueConst*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ђ
%q_next/gradients/q_next/Sum_grad/FillFill(q_next/gradients/q_next/Sum_grad/Shape_1+q_next/gradients/q_next/Sum_grad/Fill/value*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*

index_type0*
_output_shapes
: 
Ю
.q_next/gradients/q_next/Sum_grad/DynamicStitchDynamicStitch&q_next/gradients/q_next/Sum_grad/range$q_next/gradients/q_next/Sum_grad/mod&q_next/gradients/q_next/Sum_grad/Shape%q_next/gradients/q_next/Sum_grad/Fill*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
N*#
_output_shapes
:џџџџџџџџџ
Ї
*q_next/gradients/q_next/Sum_grad/Maximum/yConst*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ј
(q_next/gradients/q_next/Sum_grad/MaximumMaximum.q_next/gradients/q_next/Sum_grad/DynamicStitch*q_next/gradients/q_next/Sum_grad/Maximum/y*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*#
_output_shapes
:џџџџџџџџџ
ч
)q_next/gradients/q_next/Sum_grad/floordivFloorDiv&q_next/gradients/q_next/Sum_grad/Shape(q_next/gradients/q_next/Sum_grad/Maximum*
_output_shapes
:*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape
б
(q_next/gradients/q_next/Sum_grad/ReshapeReshape;q_next/gradients/q_next/sub_grad/tuple/control_dependency_1.q_next/gradients/q_next/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ц
%q_next/gradients/q_next/Sum_grad/TileTile(q_next/gradients/q_next/Sum_grad/Reshape)q_next/gradients/q_next/Sum_grad/floordiv*
T0*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0
p
&q_next/gradients/q_next/Mul_grad/ShapeShape
q_next/add*
T0*
out_type0*
_output_shapes
:
{
(q_next/gradients/q_next/Mul_grad/Shape_1Shapeq_next/action_taken*
T0*
out_type0*
_output_shapes
:
о
6q_next/gradients/q_next/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs&q_next/gradients/q_next/Mul_grad/Shape(q_next/gradients/q_next/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

$q_next/gradients/q_next/Mul_grad/MulMul%q_next/gradients/q_next/Sum_grad/Tileq_next/action_taken*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
$q_next/gradients/q_next/Mul_grad/SumSum$q_next/gradients/q_next/Mul_grad/Mul6q_next/gradients/q_next/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
С
(q_next/gradients/q_next/Mul_grad/ReshapeReshape$q_next/gradients/q_next/Mul_grad/Sum&q_next/gradients/q_next/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

&q_next/gradients/q_next/Mul_grad/Mul_1Mul
q_next/add%q_next/gradients/q_next/Sum_grad/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
Я
&q_next/gradients/q_next/Mul_grad/Sum_1Sum&q_next/gradients/q_next/Mul_grad/Mul_18q_next/gradients/q_next/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*q_next/gradients/q_next/Mul_grad/Reshape_1Reshape&q_next/gradients/q_next/Mul_grad/Sum_1(q_next/gradients/q_next/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

1q_next/gradients/q_next/Mul_grad/tuple/group_depsNoOp)^q_next/gradients/q_next/Mul_grad/Reshape+^q_next/gradients/q_next/Mul_grad/Reshape_1

9q_next/gradients/q_next/Mul_grad/tuple/control_dependencyIdentity(q_next/gradients/q_next/Mul_grad/Reshape2^q_next/gradients/q_next/Mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_next/gradients/q_next/Mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;q_next/gradients/q_next/Mul_grad/tuple/control_dependency_1Identity*q_next/gradients/q_next/Mul_grad/Reshape_12^q_next/gradients/q_next/Mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_next/gradients/q_next/Mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
s
&q_next/gradients/q_next/add_grad/ShapeShapeq_next/MatMul*
T0*
out_type0*
_output_shapes
:
r
(q_next/gradients/q_next/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
о
6q_next/gradients/q_next/add_grad/BroadcastGradientArgsBroadcastGradientArgs&q_next/gradients/q_next/add_grad/Shape(q_next/gradients/q_next/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
о
$q_next/gradients/q_next/add_grad/SumSum9q_next/gradients/q_next/Mul_grad/tuple/control_dependency6q_next/gradients/q_next/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(q_next/gradients/q_next/add_grad/ReshapeReshape$q_next/gradients/q_next/add_grad/Sum&q_next/gradients/q_next/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
т
&q_next/gradients/q_next/add_grad/Sum_1Sum9q_next/gradients/q_next/Mul_grad/tuple/control_dependency8q_next/gradients/q_next/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
К
*q_next/gradients/q_next/add_grad/Reshape_1Reshape&q_next/gradients/q_next/add_grad/Sum_1(q_next/gradients/q_next/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

1q_next/gradients/q_next/add_grad/tuple/group_depsNoOp)^q_next/gradients/q_next/add_grad/Reshape+^q_next/gradients/q_next/add_grad/Reshape_1

9q_next/gradients/q_next/add_grad/tuple/control_dependencyIdentity(q_next/gradients/q_next/add_grad/Reshape2^q_next/gradients/q_next/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_next/gradients/q_next/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;q_next/gradients/q_next/add_grad/tuple/control_dependency_1Identity*q_next/gradients/q_next/add_grad/Reshape_12^q_next/gradients/q_next/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_next/gradients/q_next/add_grad/Reshape_1*
_output_shapes
:
н
*q_next/gradients/q_next/MatMul_grad/MatMulMatMul9q_next/gradients/q_next/add_grad/tuple/control_dependencyq_next/weights/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
з
,q_next/gradients/q_next/MatMul_grad/MatMul_1MatMulq_next/strided_slice9q_next/gradients/q_next/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 

4q_next/gradients/q_next/MatMul_grad/tuple/group_depsNoOp+^q_next/gradients/q_next/MatMul_grad/MatMul-^q_next/gradients/q_next/MatMul_grad/MatMul_1

<q_next/gradients/q_next/MatMul_grad/tuple/control_dependencyIdentity*q_next/gradients/q_next/MatMul_grad/MatMul5^q_next/gradients/q_next/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@q_next/gradients/q_next/MatMul_grad/MatMul

>q_next/gradients/q_next/MatMul_grad/tuple/control_dependency_1Identity,q_next/gradients/q_next/MatMul_grad/MatMul_15^q_next/gradients/q_next/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@q_next/gradients/q_next/MatMul_grad/MatMul_1*
_output_shapes
:	

0q_next/gradients/q_next/strided_slice_grad/ShapeShapeq_next/rnn/transpose_1*
T0*
out_type0*
_output_shapes
:
Ш
;q_next/gradients/q_next/strided_slice_grad/StridedSliceGradStridedSliceGrad0q_next/gradients/q_next/strided_slice_grad/Shapeq_next/strided_slice/stackq_next/strided_slice/stack_1q_next/strided_slice/stack_2<q_next/gradients/q_next/MatMul_grad/tuple/control_dependency*
end_mask*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask 

>q_next/gradients/q_next/rnn/transpose_1_grad/InvertPermutationInvertPermutationq_next/rnn/concat_2*
T0*
_output_shapes
:

6q_next/gradients/q_next/rnn/transpose_1_grad/transpose	Transpose;q_next/gradients/q_next/strided_slice_grad/StridedSliceGrad>q_next/gradients/q_next/rnn/transpose_1_grad/InvertPermutation*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
Tperm0

gq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3q_next/rnn/TensorArrayq_next/rnn/while/Exit_2*)
_class
loc:@q_next/rnn/TensorArray*
sourceq_next/gradients*
_output_shapes

:: 
О
cq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityq_next/rnn/while/Exit_2h^q_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*)
_class
loc:@q_next/rnn/TensorArray*
_output_shapes
: 
Я
mq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3gq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3!q_next/rnn/TensorArrayStack/range6q_next/gradients/q_next/rnn/transpose_1_grad/transposecq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
t
q_next/gradients/zeros_like	ZerosLikeq_next/rnn/while/Exit_3*
T0*(
_output_shapes
:џџџџџџџџџ
v
q_next/gradients/zeros_like_1	ZerosLikeq_next/rnn/while/Exit_4*
T0*(
_output_shapes
:џџџџџџџџџ
М
4q_next/gradients/q_next/rnn/while/Exit_2_grad/b_exitEntermq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ќ
4q_next/gradients/q_next/rnn/while/Exit_3_grad/b_exitEnterq_next/gradients/zeros_like*
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant( 
ў
4q_next/gradients/q_next/rnn/while/Exit_4_grad/b_exitEnterq_next/gradients/zeros_like_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ф
8q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switchMerge4q_next/gradients/q_next/rnn/while/Exit_2_grad/b_exit?q_next/gradients/q_next/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
і
8q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switchMerge4q_next/gradients/q_next/rnn/while/Exit_3_grad/b_exit?q_next/gradients/q_next/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
і
8q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switchMerge4q_next/gradients/q_next/rnn/while/Exit_4_grad/b_exit?q_next/gradients/q_next/rnn/while/Switch_4_grad_1/NextIteration*
N**
_output_shapes
:џџџџџџџџџ: *
T0

5q_next/gradients/q_next/rnn/while/Merge_2_grad/SwitchSwitch8q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switchq_next/gradients/b_count_2*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 

?q_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/group_depsNoOp6^q_next/gradients/q_next/rnn/while/Merge_2_grad/Switch
К
Gq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity5q_next/gradients/q_next/rnn/while/Merge_2_grad/Switch@^q_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/group_deps*
_output_shapes
: *
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switch
О
Iq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity7q_next/gradients/q_next/rnn/while/Merge_2_grad/Switch:1@^q_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
Љ
5q_next/gradients/q_next/rnn/while/Merge_3_grad/SwitchSwitch8q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switchq_next/gradients/b_count_2*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switch*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

?q_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/group_depsNoOp6^q_next/gradients/q_next/rnn/while/Merge_3_grad/Switch
Ь
Gq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity5q_next/gradients/q_next/rnn/while/Merge_3_grad/Switch@^q_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switch
а
Iq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity7q_next/gradients/q_next/rnn/while/Merge_3_grad/Switch:1@^q_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switch
Љ
5q_next/gradients/q_next/rnn/while/Merge_4_grad/SwitchSwitch8q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switchq_next/gradients/b_count_2*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switch*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

?q_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/group_depsNoOp6^q_next/gradients/q_next/rnn/while/Merge_4_grad/Switch
Ь
Gq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity5q_next/gradients/q_next/rnn/while/Merge_4_grad/Switch@^q_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switch
а
Iq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity7q_next/gradients/q_next/rnn/while/Merge_4_grad/Switch:1@^q_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
Ѕ
3q_next/gradients/q_next/rnn/while/Enter_2_grad/ExitExitGq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
З
3q_next/gradients/q_next/rnn/while/Enter_3_grad/ExitExitGq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
З
3q_next/gradients/q_next/rnn/while/Enter_4_grad/ExitExitGq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Б
lq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterIq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1*
_output_shapes

:: *3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*
sourceq_next/gradients
м
rq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterq_next/rnn/TensorArray*
is_constant(*
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*
parallel_iterations 

hq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityIq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1m^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2
щ
\q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3lq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3gq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2hq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:џџџџџџџџџ
н
bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*.
_class$
" loc:@q_next/rnn/while/Identity_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*.
_class$
" loc:@q_next/rnn/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
в
bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterbq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(
У
hq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterq_next/rnn/while/Identity_1^q_next/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
Є
gq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2mq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^q_next/gradients/Sub*
_output_shapes
: *
	elem_type0
ю
mq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterbq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
х
cq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerB^q_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2F^q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2F^q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2h^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2L^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2X^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2Z^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1V^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2J^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2X^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2Z^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1F^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2H^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2X^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2Z^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1F^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2H^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2V^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2X^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1F^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2

[q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpJ^q_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1]^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Я
cq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentity\q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3\^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*o
_classe
caloc:@q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*(
_output_shapes
:џџџџџџџџџ

eq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityIq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1\^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
_output_shapes
: *
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switch
С
:q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like	ZerosLikeEq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
Л
@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/ConstConst*.
_class$
" loc:@q_next/rnn/while/Identity_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ќ
@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/f_accStackV2@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/Const*
	elem_type0*.
_class$
" loc:@q_next/rnn/while/Identity_3*

stack_name *
_output_shapes
:

@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/EnterEnter@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(

Fq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/Enterq_next/rnn/while/Identity_3^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
Н
6q_next/gradients/q_next/rnn/while/Select_1_grad/SelectSelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2Iq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependency_1:q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like*(
_output_shapes
:џџџџџџџџџ*
T0
Й
<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/ConstConst*0
_class&
$"loc:@q_next/rnn/while/GreaterEqual*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
і
<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/f_accStackV2<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/Const*
	elem_type0
*0
_class&
$"loc:@q_next/rnn/while/GreaterEqual*

stack_name *
_output_shapes
:

<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/EnterEnter<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(
ћ
Bq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPushV2StackPushV2<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/Enterq_next/rnn/while/GreaterEqual^q_next/gradients/Add*
T0
*
_output_shapes
:*
swap_memory( 
к
Aq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2
StackPopV2Gq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0

Ђ
Gq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2/EnterEnter<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
П
8q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1SelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2:q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_likeIq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
М
@q_next/gradients/q_next/rnn/while/Select_1_grad/tuple/group_depsNoOp7^q_next/gradients/q_next/rnn/while/Select_1_grad/Select9^q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1
Э
Hq_next/gradients/q_next/rnn/while/Select_1_grad/tuple/control_dependencyIdentity6q_next/gradients/q_next/rnn/while/Select_1_grad/SelectA^q_next/gradients/q_next/rnn/while/Select_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_1_grad/Select*(
_output_shapes
:џџџџџџџџџ
г
Jq_next/gradients/q_next/rnn/while/Select_1_grad/tuple/control_dependency_1Identity8q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1A^q_next/gradients/q_next/rnn/while/Select_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1*(
_output_shapes
:џџџџџџџџџ
С
:q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like	ZerosLikeEq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
Л
@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@q_next/rnn/while/Identity_4*
valueB :
џџџџџџџџџ
ќ
@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/f_accStackV2@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/Const*.
_class$
" loc:@q_next/rnn/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0

@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/EnterEnter@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context

Fq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/Enterq_next/rnn/while/Identity_4^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
Н
6q_next/gradients/q_next/rnn/while/Select_2_grad/SelectSelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2Iq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependency_1:q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
П
8q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1SelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2:q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_likeIq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
М
@q_next/gradients/q_next/rnn/while/Select_2_grad/tuple/group_depsNoOp7^q_next/gradients/q_next/rnn/while/Select_2_grad/Select9^q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1
Э
Hq_next/gradients/q_next/rnn/while/Select_2_grad/tuple/control_dependencyIdentity6q_next/gradients/q_next/rnn/while/Select_2_grad/SelectA^q_next/gradients/q_next/rnn/while/Select_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_2_grad/Select*(
_output_shapes
:џџџџџџџџџ
г
Jq_next/gradients/q_next/rnn/while/Select_2_grad/tuple/control_dependency_1Identity8q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1A^q_next/gradients/q_next/rnn/while/Select_2_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1
Я
8q_next/gradients/q_next/rnn/while/Select_grad/zeros_like	ZerosLike>q_next/gradients/q_next/rnn/while/Select_grad/zeros_like/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
T0
ћ
>q_next/gradients/q_next/rnn/while/Select_grad/zeros_like/EnterEnterq_next/rnn/zeros*
T0*
is_constant(*
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
г
4q_next/gradients/q_next/rnn/while/Select_grad/SelectSelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2cq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency8q_next/gradients/q_next/rnn/while/Select_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
е
6q_next/gradients/q_next/rnn/while/Select_grad/Select_1SelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV28q_next/gradients/q_next/rnn/while/Select_grad/zeros_likecq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
>q_next/gradients/q_next/rnn/while/Select_grad/tuple/group_depsNoOp5^q_next/gradients/q_next/rnn/while/Select_grad/Select7^q_next/gradients/q_next/rnn/while/Select_grad/Select_1
Х
Fq_next/gradients/q_next/rnn/while/Select_grad/tuple/control_dependencyIdentity4q_next/gradients/q_next/rnn/while/Select_grad/Select?^q_next/gradients/q_next/rnn/while/Select_grad/tuple/group_deps*
T0*G
_class=
;9loc:@q_next/gradients/q_next/rnn/while/Select_grad/Select*(
_output_shapes
:џџџџџџџџџ
Ы
Hq_next/gradients/q_next/rnn/while/Select_grad/tuple/control_dependency_1Identity6q_next/gradients/q_next/rnn/while/Select_grad/Select_1?^q_next/gradients/q_next/rnn/while/Select_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_grad/Select_1*(
_output_shapes
:џџџџџџџџџ

9q_next/gradients/q_next/rnn/while/Select/Enter_grad/ShapeShapeq_next/rnn/zeros*
T0*
out_type0*
_output_shapes
:

?q_next/gradients/q_next/rnn/while/Select/Enter_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

9q_next/gradients/q_next/rnn/while/Select/Enter_grad/zerosFill9q_next/gradients/q_next/rnn/while/Select/Enter_grad/Shape?q_next/gradients/q_next/rnn/while/Select/Enter_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0

9q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_accEnter9q_next/gradients/q_next/rnn/while/Select/Enter_grad/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

;q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_acc_1Merge9q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_accAq_next/gradients/q_next/rnn/while/Select/Enter_grad/NextIteration*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
ф
:q_next/gradients/q_next/rnn/while/Select/Enter_grad/SwitchSwitch;q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_acc_1q_next/gradients/b_count_2*
T0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
ї
7q_next/gradients/q_next/rnn/while/Select/Enter_grad/AddAdd<q_next/gradients/q_next/rnn/while/Select/Enter_grad/Switch:1Fq_next/gradients/q_next/rnn/while/Select_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
О
Aq_next/gradients/q_next/rnn/while/Select/Enter_grad/NextIterationNextIteration7q_next/gradients/q_next/rnn/while/Select/Enter_grad/Add*
T0*(
_output_shapes
:џџџџџџџџџ
В
;q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_acc_2Exit:q_next/gradients/q_next/rnn/while/Select/Enter_grad/Switch*
T0*(
_output_shapes
:џџџџџџџџџ
М
q_next/gradients/AddNAddNJq_next/gradients/q_next/rnn/while/Select_2_grad/tuple/control_dependency_1Hq_next/gradients/q_next/rnn/while/Select_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1*
N*(
_output_shapes
:џџџџџџџџџ
 
<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/ShapeShape$q_next/rnn/while/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape_1Shape!q_next/rnn/while/lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
ж
Lq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsWq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ю
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:
В
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ш
Xq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
]q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ђ
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
Ж
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1EnterTq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ю
Zq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1>q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape_1^q_next/gradients/Add*
_output_shapes
:*
swap_memory( *
T0

Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
в
_q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
в
:q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/MulMulq_next/gradients/AddNEq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0
С
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/ConstConst*4
_class*
(&loc:@q_next/rnn/while/lstm_cell/Tanh_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/f_accStackV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/Const*
	elem_type0*4
_class*
(&loc:@q_next/rnn/while/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context

Fq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/Enter!q_next/rnn/while/lstm_cell/Tanh_1^q_next/gradients/Add*(
_output_shapes
:џџџџџџџџџ*
swap_memory( *
T0
ђ
Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/SumSum:q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/MulLq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/ReshapeReshape:q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/SumWq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
ж
<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1MulGq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2q_next/gradients/AddN*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*7
_class-
+)loc:@q_next/rnn/while/lstm_cell/Sigmoid_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/Const*7
_class-
+)loc:@q_next/rnn/while/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0

Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterBq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context

Hq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter$q_next/rnn/while/lstm_cell/Sigmoid_2^q_next/gradients/Add*(
_output_shapes
:џџџџџџџџџ*
swap_memory( *
T0
і
Gq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ў
Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterBq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Sum_1Sum<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1Nq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѕ
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape_1Reshape<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Sum_1Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
г
Gq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/ReshapeA^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape_1
ы
Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentity>q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/ReshapeH^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1Identity@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape_1H^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape_1
и
?q_next/gradients/q_next/rnn/while/Switch_2_grad_1/NextIterationNextIterationeq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
}
,q_next/gradients/q_next/rnn/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
к
*q_next/gradients/q_next/rnn/zeros_grad/SumSum;q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_acc_2,q_next/gradients/q_next/rnn/zeros_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Ђ
Fq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradGq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

@q_next/gradients/q_next/rnn/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradEq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
q_next/gradients/AddN_1AddNJq_next/gradients/q_next/rnn/while/Select_1_grad/tuple/control_dependency_1@q_next/gradients/q_next/rnn/while/lstm_cell/Tanh_1_grad/TanhGrad*
N*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1

<q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/ShapeShapeq_next/rnn/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:

>q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape_1Shape q_next/rnn/while/lstm_cell/mul_1*
_output_shapes
:*
T0*
out_type0
ж
Lq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsWq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2Yq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ю
Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:
В
Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ш
Xq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter<q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
Ю
]q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant(
ђ
Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:
Ж
Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1EnterTq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(
Ю
Zq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1>q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape_1^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_next/gradients/Sub*
	elem_type0*
_output_shapes
:
в
_q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant(
ш
:q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/SumSumq_next/gradients/AddN_1Lq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/ReshapeReshape:q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/SumWq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
ь
<q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Sum_1Sumq_next/gradients/AddN_1Nq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape_1Reshape<q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Sum_1Yq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Gq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/ReshapeA^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape_1
ы
Oq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentity>q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/ReshapeH^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape
ё
Qq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1Identity@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape_1H^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/ShapeShape"q_next/rnn/while/lstm_cell/Sigmoid*
_output_shapes
:*
T0*
out_type0

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape_1Shapeq_next/rnn/while/Identity_3*
T0*
out_type0*
_output_shapes
:
а
Jq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsUq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*M
_classC
A?loc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*M
_classC
A?loc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
Ў
Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterPq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Т
Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Uq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2[q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
Ъ
[q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterPq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ю
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
В
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1EnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ш
Xq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1<q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape_1^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2]q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
Ю
]q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

8q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/MulMulOq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyEq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ

8q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/SumSum8q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/MulJq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/ReshapeReshape8q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/SumUq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1MulEq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2Oq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Т
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/ConstConst*5
_class+
)'loc:@q_next/rnn/while/lstm_cell/Sigmoid*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/f_accStackV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*5
_class+
)'loc:@q_next/rnn/while/lstm_cell/Sigmoid

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(

Fq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/Enter"q_next/rnn/while/lstm_cell/Sigmoid^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Sum_1Sum:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1Lq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape_1Reshape:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Sum_1Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Э
Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/group_depsNoOp=^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape?^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape_1
у
Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/control_dependencyIdentity<q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/ReshapeF^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape
щ
Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1Identity>q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape_1F^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
 
<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/ShapeShape$q_next/rnn/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape_1Shapeq_next/rnn/while/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
ж
Lq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsWq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ю
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape
В
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(
Ш
Xq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
Ю
]q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant(
ђ
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
Ж
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1EnterTq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ю
Zq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1>q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape_1^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
в
_q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/MulMulQq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
П
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *2
_class(
&$loc:@q_next/rnn/while/lstm_cell/Tanh*
valueB :
џџџџџџџџџ

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/f_accStackV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/Const*
	elem_type0*2
_class(
&$loc:@q_next/rnn/while/lstm_cell/Tanh*

stack_name *
_output_shapes
:

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(

Fq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/Enterq_next/rnn/while/lstm_cell/Tanh^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant(

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/SumSum:q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/MulLq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/ReshapeReshape:q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/SumWq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1MulGq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *7
_class-
+)loc:@q_next/rnn/while/lstm_cell/Sigmoid_1*
valueB :
џџџџџџџџџ

Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*7
_class-
+)loc:@q_next/rnn/while/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:

Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterBq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(

Hq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter$q_next/rnn/while/lstm_cell/Sigmoid_1^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
і
Gq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ў
Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterBq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Sum_1Sum<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1Nq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѕ
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape_1Reshape<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Sum_1Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
г
Gq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/ReshapeA^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape_1
ы
Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentity>q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/ReshapeH^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape
ё
Qq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1Identity@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape_1H^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

Dq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradEq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
С
q_next/gradients/AddN_2AddNHq_next/gradients/q_next/rnn/while/Select_1_grad/tuple/control_dependencyOq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_1_grad/Select
Ђ
Fq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradGq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

>q_next/gradients/q_next/rnn/while/lstm_cell/Tanh_grad/TanhGradTanhGradEq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:џџџџџџџџџ*
T0

:q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/ShapeShape"q_next/rnn/while/lstm_cell/split:2*
_output_shapes
:*
T0*
out_type0

<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape_1Const^q_next/gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
Е
Jq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsUq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*M
_classC
A?loc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*
	elem_type0*M
_classC
A?loc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape*

stack_name *
_output_shapes
:
Ў
Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterPq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Т
Vq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Uq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2[q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0*
_output_shapes
:
Ъ
[q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterPq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

8q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/SumSumDq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradJq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/ReshapeReshape8q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/SumUq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

:q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Sum_1SumDq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradLq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ђ
>q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape_1Reshape:q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Sum_1<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Э
Eq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/group_depsNoOp=^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape?^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape_1
у
Mq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/control_dependencyIdentity<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/ReshapeF^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
з
Oq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/control_dependency_1Identity>q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape_1F^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 

?q_next/gradients/q_next/rnn/while/Switch_3_grad_1/NextIterationNextIterationq_next/gradients/AddN_2*
T0*(
_output_shapes
:џџџџџџџџџ
ѕ
=q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concatConcatV2Fq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGrad>q_next/gradients/q_next/rnn/while/lstm_cell/Tanh_grad/TanhGradMq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/control_dependencyFq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradCq_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat/Const*

Tidx0*
T0*
N*(
_output_shapes
:џџџџџџџџџ

Cq_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat/ConstConst^q_next/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
Я
Dq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad=q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
и
Iq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpE^q_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad>^q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat
э
Qq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity=q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concatJ^q_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat*(
_output_shapes
:џџџџџџџџџ
№
Sq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityDq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradJ^q_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@q_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
К
>q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMulMatMulQq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyDq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ *
transpose_a( 

Dq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul/EnterEnter q_next/rnn/lstm_cell/kernel/read*
parallel_iterations * 
_output_shapes
:
 *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant(
Л
@q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1MatMulKq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
 *
transpose_a(*
transpose_b( 
Ч
Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*4
_class*
(&loc:@q_next/rnn/while/lstm_cell/concat*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*4
_class*
(&loc:@q_next/rnn/while/lstm_cell/concat

Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterFq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ѓ
Lq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter!q_next/rnn/while/lstm_cell/concat^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ *
swap_memory( 
ў
Kq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ *
	elem_type0
Ж
Qq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterFq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
д
Hq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMulA^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1
э
Pq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentity>q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMulI^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ 
ы
Rq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1Identity@q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1I^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
 

Dq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes	
:
Њ
Fq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterDq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
parallel_iterations *
_output_shapes	
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant( 

Fq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeFq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Lq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
р
Eq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchFq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2q_next/gradients/b_count_2*"
_output_shapes
::*
T0

Bq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/AddAddGq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch:1Sq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ч
Lq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationBq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
Л
Fq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitEq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:

=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ConstConst^q_next/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

<q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/RankConst^q_next/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
х
;q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/modFloorMod=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Const<q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0

=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeShape"q_next/rnn/while/TensorArrayReadV3*
_output_shapes
:*
T0*
out_type0

>q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeNShapeNIq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2Eq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2*
T0*
out_type0*
N* 
_output_shapes
::
Ц
Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/ConstConst*5
_class+
)'loc:@q_next/rnn/while/TensorArrayReadV3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/Const*5
_class+
)'loc:@q_next/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0

Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/EnterEnterDq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
 
Jq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/Enter"q_next/rnn/while/TensorArrayReadV3^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ *
swap_memory( 
њ
Iq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Oq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ *
	elem_type0
В
Oq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterDq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
О
Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ConcatOffsetConcatOffset;q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/mod>q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN@q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
ц
=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/SliceSlicePq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyDq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ConcatOffset>q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ь
?q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice_1SlicePq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyFq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ConcatOffset:1@q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
в
Hq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/group_depsNoOp>^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice@^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice_1
ы
Pq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyIdentity=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/SliceI^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice*(
_output_shapes
:џџџџџџџџџ 
ё
Rq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1Identity?q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice_1I^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*R
_classH
FDloc:@q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice_1*(
_output_shapes
:џџџџџџџџџ

Cq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
 *    *
dtype0* 
_output_shapes
:
 
­
Eq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterCq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations * 
_output_shapes
:
 *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant( 

Eq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeEq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1Kq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
 : 
ш
Dq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchEq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2q_next/gradients/b_count_2*
T0*,
_output_shapes
:
 :
 

Aq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/AddAddFq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/Switch:1Rq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
 
Ъ
Kq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationAq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
 *
T0
О
Eq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitDq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
 *
T0
Х
Zq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3`q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterbq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^q_next/gradients/Sub*;
_class1
/-loc:@q_next/rnn/while/TensorArrayReadV3/Enter*
sourceq_next/gradients*
_output_shapes

:: 
д
`q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterq_next/rnn/TensorArray_1*
is_constant(*
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*;
_class1
/-loc:@q_next/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
џ
bq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterEq_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*;
_class1
/-loc:@q_next/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(

Vq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitybq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1[^q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*;
_class1
/-loc:@q_next/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 

\q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Zq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3gq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Pq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyVq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
Ф
q_next/gradients/AddN_3AddNHq_next/gradients/q_next/rnn/while/Select_2_grad/tuple/control_dependencyRq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_2_grad/Select

Fq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
Hq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterFq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

Hq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeHq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Nq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
к
Gq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchHq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2q_next/gradients/b_count_2*
T0*
_output_shapes
: : 

Dq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddIq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1\q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ц
Nq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationDq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
К
Hq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitGq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 

?q_next/gradients/q_next/rnn/while/Switch_4_grad_1/NextIterationNextIterationq_next/gradients/AddN_3*
T0*(
_output_shapes
:џџџџџџџџџ
п
}q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3q_next/rnn/TensorArray_1Hq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*+
_class!
loc:@q_next/rnn/TensorArray_1*
sourceq_next/gradients*
_output_shapes

:: 

yq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityHq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3~^q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@q_next/rnn/TensorArray_1*
_output_shapes
: 

oq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3}q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3#q_next/rnn/TensorArrayUnstack/rangeyq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ 
Б
lq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpp^q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3I^q_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
Ѕ
tq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityoq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3m^q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*
_classx
vtloc:@q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ 
Й
vq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityHq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3m^q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@q_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 

<q_next/gradients/q_next/rnn/transpose_grad/InvertPermutationInvertPermutationq_next/rnn/concat*
_output_shapes
:*
T0
Т
4q_next/gradients/q_next/rnn/transpose_grad/transpose	Transposetq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency<q_next/gradients/q_next/rnn/transpose_grad/InvertPermutation*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
Tperm0*
T0
z
,q_next/gradients/q_next/Reshape_1_grad/ShapeShapeq_next/Reshape*
_output_shapes
:*
T0*
out_type0
о
.q_next/gradients/q_next/Reshape_1_grad/ReshapeReshape4q_next/gradients/q_next/rnn/transpose_grad/transpose,q_next/gradients/q_next/Reshape_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ 
w
*q_next/gradients/q_next/Reshape_grad/ShapeShapeq_next/Relu_1*
_output_shapes
:*
T0*
out_type0
л
,q_next/gradients/q_next/Reshape_grad/ReshapeReshape.q_next/gradients/q_next/Reshape_1_grad/Reshape*q_next/gradients/q_next/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ		 
Џ
,q_next/gradients/q_next/Relu_1_grad/ReluGradReluGrad,q_next/gradients/q_next/Reshape_grad/Reshapeq_next/Relu_1*
T0*/
_output_shapes
:џџџџџџџџџ		 
Џ
6q_next/gradients/q_next/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad,q_next/gradients/q_next/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
Ћ
;q_next/gradients/q_next/conv2/BiasAdd_grad/tuple/group_depsNoOp-^q_next/gradients/q_next/Relu_1_grad/ReluGrad7^q_next/gradients/q_next/conv2/BiasAdd_grad/BiasAddGrad
Ж
Cq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependencyIdentity,q_next/gradients/q_next/Relu_1_grad/ReluGrad<^q_next/gradients/q_next/conv2/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@q_next/gradients/q_next/Relu_1_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ		 
З
Eq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependency_1Identity6q_next/gradients/q_next/conv2/BiasAdd_grad/BiasAddGrad<^q_next/gradients/q_next/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*I
_class?
=;loc:@q_next/gradients/q_next/conv2/BiasAdd_grad/BiasAddGrad
Ѕ
0q_next/gradients/q_next/conv2/Conv2D_grad/ShapeNShapeNq_next/Reluq_next/conv2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::

/q_next/gradients/q_next/conv2/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Љ
=q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput0q_next/gradients/q_next/conv2/Conv2D_grad/ShapeNq_next/conv2/kernel/readCq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations

љ
>q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterq_next/Relu/q_next/gradients/q_next/conv2/Conv2D_grad/ConstCq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
: *
	dilations
*
T0
У
:q_next/gradients/q_next/conv2/Conv2D_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropFilter>^q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropInput
ж
Bq_next/gradients/q_next/conv2/Conv2D_grad/tuple/control_dependencyIdentity=q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropInput;^q_next/gradients/q_next/conv2/Conv2D_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ
б
Dq_next/gradients/q_next/conv2/Conv2D_grad/tuple/control_dependency_1Identity>q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropFilter;^q_next/gradients/q_next/conv2/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
С
*q_next/gradients/q_next/Relu_grad/ReluGradReluGradBq_next/gradients/q_next/conv2/Conv2D_grad/tuple/control_dependencyq_next/Relu*
T0*/
_output_shapes
:џџџџџџџџџ
­
6q_next/gradients/q_next/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad*q_next/gradients/q_next/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0
Љ
;q_next/gradients/q_next/conv1/BiasAdd_grad/tuple/group_depsNoOp+^q_next/gradients/q_next/Relu_grad/ReluGrad7^q_next/gradients/q_next/conv1/BiasAdd_grad/BiasAddGrad
В
Cq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependencyIdentity*q_next/gradients/q_next/Relu_grad/ReluGrad<^q_next/gradients/q_next/conv1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_next/gradients/q_next/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ
З
Eq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependency_1Identity6q_next/gradients/q_next/conv1/BiasAdd_grad/BiasAddGrad<^q_next/gradients/q_next/conv1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_next/gradients/q_next/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ї
0q_next/gradients/q_next/conv1/Conv2D_grad/ShapeNShapeNq_next/statesq_next/conv1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::

/q_next/gradients/q_next/conv1/Conv2D_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
Љ
=q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput0q_next/gradients/q_next/conv1/Conv2D_grad/ShapeNq_next/conv1/kernel/readCq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
ћ
>q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterq_next/states/q_next/gradients/q_next/conv1/Conv2D_grad/ConstCq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
У
:q_next/gradients/q_next/conv1/Conv2D_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropFilter>^q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropInput
ж
Bq_next/gradients/q_next/conv1/Conv2D_grad/tuple/control_dependencyIdentity=q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropInput;^q_next/gradients/q_next/conv1/Conv2D_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџTT
б
Dq_next/gradients/q_next/conv1/Conv2D_grad/tuple/control_dependency_1Identity>q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropFilter;^q_next/gradients/q_next/conv1/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

 q_next/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: * 
_class
loc:@q_next/biases*
valueB
 *fff?

q_next/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@q_next/biases*
	container 
Х
q_next/beta1_power/AssignAssignq_next/beta1_power q_next/beta1_power/initial_value*
use_locking(*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: 
z
q_next/beta1_power/readIdentityq_next/beta1_power*
T0* 
_class
loc:@q_next/biases*
_output_shapes
: 

 q_next/beta2_power/initial_valueConst* 
_class
loc:@q_next/biases*
valueB
 *wО?*
dtype0*
_output_shapes
: 

q_next/beta2_power
VariableV2* 
_class
loc:@q_next/biases*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Х
q_next/beta2_power/AssignAssignq_next/beta2_power q_next/beta2_power/initial_value*
use_locking(*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: 
z
q_next/beta2_power/readIdentityq_next/beta2_power*
_output_shapes
: *
T0* 
_class
loc:@q_next/biases
Т
Aq_next/q_next/conv1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_next/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Є
7q_next/q_next/conv1/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@q_next/conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
 
1q_next/q_next/conv1/kernel/Adam/Initializer/zerosFillAq_next/q_next/conv1/kernel/Adam/Initializer/zeros/shape_as_tensor7q_next/q_next/conv1/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@q_next/conv1/kernel*

index_type0*&
_output_shapes
:
Ы
q_next/q_next/conv1/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@q_next/conv1/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:

&q_next/q_next/conv1/kernel/Adam/AssignAssignq_next/q_next/conv1/kernel/Adam1q_next/q_next/conv1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@q_next/conv1/kernel*
validate_shape(*&
_output_shapes
:
Њ
$q_next/q_next/conv1/kernel/Adam/readIdentityq_next/q_next/conv1/kernel/Adam*&
_output_shapes
:*
T0*&
_class
loc:@q_next/conv1/kernel
Ф
Cq_next/q_next/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_next/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
І
9q_next/q_next/conv1/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@q_next/conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
І
3q_next/q_next/conv1/kernel/Adam_1/Initializer/zerosFillCq_next/q_next/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensor9q_next/q_next/conv1/kernel/Adam_1/Initializer/zeros/Const*&
_output_shapes
:*
T0*&
_class
loc:@q_next/conv1/kernel*

index_type0
Э
!q_next/q_next/conv1/kernel/Adam_1
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *&
_class
loc:@q_next/conv1/kernel*
	container *
shape:

(q_next/q_next/conv1/kernel/Adam_1/AssignAssign!q_next/q_next/conv1/kernel/Adam_13q_next/q_next/conv1/kernel/Adam_1/Initializer/zeros*
T0*&
_class
loc:@q_next/conv1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
Ў
&q_next/q_next/conv1/kernel/Adam_1/readIdentity!q_next/q_next/conv1/kernel/Adam_1*
T0*&
_class
loc:@q_next/conv1/kernel*&
_output_shapes
:
Ђ
/q_next/q_next/conv1/bias/Adam/Initializer/zerosConst*$
_class
loc:@q_next/conv1/bias*
valueB*    *
dtype0*
_output_shapes
:
Џ
q_next/q_next/conv1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@q_next/conv1/bias*
	container *
shape:
ђ
$q_next/q_next/conv1/bias/Adam/AssignAssignq_next/q_next/conv1/bias/Adam/q_next/q_next/conv1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_next/conv1/bias

"q_next/q_next/conv1/bias/Adam/readIdentityq_next/q_next/conv1/bias/Adam*
T0*$
_class
loc:@q_next/conv1/bias*
_output_shapes
:
Є
1q_next/q_next/conv1/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@q_next/conv1/bias*
valueB*    *
dtype0*
_output_shapes
:
Б
q_next/q_next/conv1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@q_next/conv1/bias*
	container *
shape:
ј
&q_next/q_next/conv1/bias/Adam_1/AssignAssignq_next/q_next/conv1/bias/Adam_11q_next/q_next/conv1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_next/conv1/bias

$q_next/q_next/conv1/bias/Adam_1/readIdentityq_next/q_next/conv1/bias/Adam_1*
T0*$
_class
loc:@q_next/conv1/bias*
_output_shapes
:
Т
Aq_next/q_next/conv2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_next/conv2/kernel*%
valueB"             *
dtype0*
_output_shapes
:
Є
7q_next/q_next/conv2/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@q_next/conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
 
1q_next/q_next/conv2/kernel/Adam/Initializer/zerosFillAq_next/q_next/conv2/kernel/Adam/Initializer/zeros/shape_as_tensor7q_next/q_next/conv2/kernel/Adam/Initializer/zeros/Const*&
_output_shapes
: *
T0*&
_class
loc:@q_next/conv2/kernel*

index_type0
Ы
q_next/q_next/conv2/kernel/Adam
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_next/conv2/kernel*
	container *
shape: 

&q_next/q_next/conv2/kernel/Adam/AssignAssignq_next/q_next/conv2/kernel/Adam1q_next/q_next/conv2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel*
validate_shape(*&
_output_shapes
: 
Њ
$q_next/q_next/conv2/kernel/Adam/readIdentityq_next/q_next/conv2/kernel/Adam*
T0*&
_class
loc:@q_next/conv2/kernel*&
_output_shapes
: 
Ф
Cq_next/q_next/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*&
_class
loc:@q_next/conv2/kernel*%
valueB"             
І
9q_next/q_next/conv2/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@q_next/conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
І
3q_next/q_next/conv2/kernel/Adam_1/Initializer/zerosFillCq_next/q_next/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensor9q_next/q_next/conv2/kernel/Adam_1/Initializer/zeros/Const*&
_output_shapes
: *
T0*&
_class
loc:@q_next/conv2/kernel*

index_type0
Э
!q_next/q_next/conv2/kernel/Adam_1
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_next/conv2/kernel*
	container *
shape: 

(q_next/q_next/conv2/kernel/Adam_1/AssignAssign!q_next/q_next/conv2/kernel/Adam_13q_next/q_next/conv2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel*
validate_shape(*&
_output_shapes
: 
Ў
&q_next/q_next/conv2/kernel/Adam_1/readIdentity!q_next/q_next/conv2/kernel/Adam_1*
T0*&
_class
loc:@q_next/conv2/kernel*&
_output_shapes
: 
Ђ
/q_next/q_next/conv2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
: *$
_class
loc:@q_next/conv2/bias*
valueB *    
Џ
q_next/q_next/conv2/bias/Adam
VariableV2*
shared_name *$
_class
loc:@q_next/conv2/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ђ
$q_next/q_next/conv2/bias/Adam/AssignAssignq_next/q_next/conv2/bias/Adam/q_next/q_next/conv2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias

"q_next/q_next/conv2/bias/Adam/readIdentityq_next/q_next/conv2/bias/Adam*
T0*$
_class
loc:@q_next/conv2/bias*
_output_shapes
: 
Є
1q_next/q_next/conv2/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@q_next/conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Б
q_next/q_next/conv2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@q_next/conv2/bias*
	container *
shape: 
ј
&q_next/q_next/conv2/bias/Adam_1/AssignAssignq_next/q_next/conv2/bias/Adam_11q_next/q_next/conv2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias

$q_next/q_next/conv2/bias/Adam_1/readIdentityq_next/q_next/conv2/bias/Adam_1*
T0*$
_class
loc:@q_next/conv2/bias*
_output_shapes
: 
Ъ
Iq_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Д
?q_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
К
9q_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zerosFillIq_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensor?q_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*

index_type0* 
_output_shapes
:
 
Я
'q_next/q_next/rnn/lstm_cell/kernel/Adam
VariableV2*
shared_name *.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
	container *
shape:
 *
dtype0* 
_output_shapes
:
 
 
.q_next/q_next/rnn/lstm_cell/kernel/Adam/AssignAssign'q_next/q_next/rnn/lstm_cell/kernel/Adam9q_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
М
,q_next/q_next/rnn/lstm_cell/kernel/Adam/readIdentity'q_next/q_next/rnn/lstm_cell/kernel/Adam*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel* 
_output_shapes
:
 
Ь
Kq_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ж
Aq_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
;q_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zerosFillKq_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorAq_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*

index_type0* 
_output_shapes
:
 
б
)q_next/q_next/rnn/lstm_cell/kernel/Adam_1
VariableV2*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
	container *
shape:
 *
dtype0* 
_output_shapes
:
 *
shared_name 
І
0q_next/q_next/rnn/lstm_cell/kernel/Adam_1/AssignAssign)q_next/q_next/rnn/lstm_cell/kernel/Adam_1;q_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
Р
.q_next/q_next/rnn/lstm_cell/kernel/Adam_1/readIdentity)q_next/q_next/rnn/lstm_cell/kernel/Adam_1*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel* 
_output_shapes
:
 
Р
Gq_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
А
=q_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros/ConstConst*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
­
7q_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zerosFillGq_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensor=q_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
С
%q_next/q_next/rnn/lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
	container *
shape:

,q_next/q_next/rnn/lstm_cell/bias/Adam/AssignAssign%q_next/q_next/rnn/lstm_cell/bias/Adam7q_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias
Б
*q_next/q_next/rnn/lstm_cell/bias/Adam/readIdentity%q_next/q_next/rnn/lstm_cell/bias/Adam*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
_output_shapes	
:
Т
Iq_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
В
?q_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/ConstConst*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
9q_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zerosFillIq_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensor?q_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*

index_type0
У
'q_next/q_next/rnn/lstm_cell/bias/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
	container *
shape:*
dtype0*
_output_shapes	
:

.q_next/q_next/rnn/lstm_cell/bias/Adam_1/AssignAssign'q_next/q_next/rnn/lstm_cell/bias/Adam_19q_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
Е
,q_next/q_next/rnn/lstm_cell/bias/Adam_1/readIdentity'q_next/q_next/rnn/lstm_cell/bias/Adam_1*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
_output_shapes	
:
А
<q_next/q_next/weights/Adam/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@q_next/weights*
valueB"      *
dtype0*
_output_shapes
:

2q_next/q_next/weights/Adam/Initializer/zeros/ConstConst*!
_class
loc:@q_next/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

,q_next/q_next/weights/Adam/Initializer/zerosFill<q_next/q_next/weights/Adam/Initializer/zeros/shape_as_tensor2q_next/q_next/weights/Adam/Initializer/zeros/Const*
T0*!
_class
loc:@q_next/weights*

index_type0*
_output_shapes
:	
Г
q_next/q_next/weights/Adam
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *!
_class
loc:@q_next/weights
ы
!q_next/q_next/weights/Adam/AssignAssignq_next/q_next/weights/Adam,q_next/q_next/weights/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	

q_next/q_next/weights/Adam/readIdentityq_next/q_next/weights/Adam*
_output_shapes
:	*
T0*!
_class
loc:@q_next/weights
В
>q_next/q_next/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@q_next/weights*
valueB"      *
dtype0*
_output_shapes
:

4q_next/q_next/weights/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@q_next/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

.q_next/q_next/weights/Adam_1/Initializer/zerosFill>q_next/q_next/weights/Adam_1/Initializer/zeros/shape_as_tensor4q_next/q_next/weights/Adam_1/Initializer/zeros/Const*
T0*!
_class
loc:@q_next/weights*

index_type0*
_output_shapes
:	
Е
q_next/q_next/weights/Adam_1
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *!
_class
loc:@q_next/weights*
	container 
ё
#q_next/q_next/weights/Adam_1/AssignAssignq_next/q_next/weights/Adam_1.q_next/q_next/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	

!q_next/q_next/weights/Adam_1/readIdentityq_next/q_next/weights/Adam_1*
T0*!
_class
loc:@q_next/weights*
_output_shapes
:	

+q_next/q_next/biases/Adam/Initializer/zerosConst* 
_class
loc:@q_next/biases*
valueB*    *
dtype0*
_output_shapes
:
Ї
q_next/q_next/biases/Adam
VariableV2*
shared_name * 
_class
loc:@q_next/biases*
	container *
shape:*
dtype0*
_output_shapes
:
т
 q_next/q_next/biases/Adam/AssignAssignq_next/q_next/biases/Adam+q_next/q_next/biases/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
:

q_next/q_next/biases/Adam/readIdentityq_next/q_next/biases/Adam*
T0* 
_class
loc:@q_next/biases*
_output_shapes
:

-q_next/q_next/biases/Adam_1/Initializer/zerosConst* 
_class
loc:@q_next/biases*
valueB*    *
dtype0*
_output_shapes
:
Љ
q_next/q_next/biases/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@q_next/biases*
	container 
ш
"q_next/q_next/biases/Adam_1/AssignAssignq_next/q_next/biases/Adam_1-q_next/q_next/biases/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
:

 q_next/q_next/biases/Adam_1/readIdentityq_next/q_next/biases/Adam_1*
T0* 
_class
loc:@q_next/biases*
_output_shapes
:
^
q_next/Adam/learning_rateConst*
valueB
 *o9*
dtype0*
_output_shapes
: 
V
q_next/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
V
q_next/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
X
q_next/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
ф
0q_next/Adam/update_q_next/conv1/kernel/ApplyAdam	ApplyAdamq_next/conv1/kernelq_next/q_next/conv1/kernel/Adam!q_next/q_next/conv1/kernel/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonDq_next/gradients/q_next/conv1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@q_next/conv1/kernel*
use_nesterov( *&
_output_shapes
:
Я
.q_next/Adam/update_q_next/conv1/bias/ApplyAdam	ApplyAdamq_next/conv1/biasq_next/q_next/conv1/bias/Adamq_next/q_next/conv1/bias/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonEq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*$
_class
loc:@q_next/conv1/bias
ф
0q_next/Adam/update_q_next/conv2/kernel/ApplyAdam	ApplyAdamq_next/conv2/kernelq_next/q_next/conv2/kernel/Adam!q_next/q_next/conv2/kernel/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonDq_next/gradients/q_next/conv2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@q_next/conv2/kernel*
use_nesterov( *&
_output_shapes
: 
Я
.q_next/Adam/update_q_next/conv2/bias/ApplyAdam	ApplyAdamq_next/conv2/biasq_next/q_next/conv2/bias/Adamq_next/q_next/conv2/bias/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonEq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@q_next/conv2/bias*
use_nesterov( *
_output_shapes
: 

8q_next/Adam/update_q_next/rnn/lstm_cell/kernel/ApplyAdam	ApplyAdamq_next/rnn/lstm_cell/kernel'q_next/q_next/rnn/lstm_cell/kernel/Adam)q_next/q_next/rnn/lstm_cell/kernel/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonEq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
use_nesterov( * 
_output_shapes
:
 *
use_locking( 
љ
6q_next/Adam/update_q_next/rnn/lstm_cell/bias/ApplyAdam	ApplyAdamq_next/rnn/lstm_cell/bias%q_next/q_next/rnn/lstm_cell/bias/Adam'q_next/q_next/rnn/lstm_cell/bias/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonFq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias
О
+q_next/Adam/update_q_next/weights/ApplyAdam	ApplyAdamq_next/weightsq_next/q_next/weights/Adamq_next/q_next/weights/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilon>q_next/gradients/q_next/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@q_next/weights*
use_nesterov( *
_output_shapes
:	
Б
*q_next/Adam/update_q_next/biases/ApplyAdam	ApplyAdamq_next/biasesq_next/q_next/biases/Adamq_next/q_next/biases/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilon;q_next/gradients/q_next/add_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@q_next/biases*
use_nesterov( *
_output_shapes
:

q_next/Adam/mulMulq_next/beta1_power/readq_next/Adam/beta1+^q_next/Adam/update_q_next/biases/ApplyAdam/^q_next/Adam/update_q_next/conv1/bias/ApplyAdam1^q_next/Adam/update_q_next/conv1/kernel/ApplyAdam/^q_next/Adam/update_q_next/conv2/bias/ApplyAdam1^q_next/Adam/update_q_next/conv2/kernel/ApplyAdam7^q_next/Adam/update_q_next/rnn/lstm_cell/bias/ApplyAdam9^q_next/Adam/update_q_next/rnn/lstm_cell/kernel/ApplyAdam,^q_next/Adam/update_q_next/weights/ApplyAdam*
T0* 
_class
loc:@q_next/biases*
_output_shapes
: 
­
q_next/Adam/AssignAssignq_next/beta1_powerq_next/Adam/mul*
use_locking( *
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: 

q_next/Adam/mul_1Mulq_next/beta2_power/readq_next/Adam/beta2+^q_next/Adam/update_q_next/biases/ApplyAdam/^q_next/Adam/update_q_next/conv1/bias/ApplyAdam1^q_next/Adam/update_q_next/conv1/kernel/ApplyAdam/^q_next/Adam/update_q_next/conv2/bias/ApplyAdam1^q_next/Adam/update_q_next/conv2/kernel/ApplyAdam7^q_next/Adam/update_q_next/rnn/lstm_cell/bias/ApplyAdam9^q_next/Adam/update_q_next/rnn/lstm_cell/kernel/ApplyAdam,^q_next/Adam/update_q_next/weights/ApplyAdam*
T0* 
_class
loc:@q_next/biases*
_output_shapes
: 
Б
q_next/Adam/Assign_1Assignq_next/beta2_powerq_next/Adam/mul_1*
use_locking( *
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: 
ж
q_next/AdamNoOp^q_next/Adam/Assign^q_next/Adam/Assign_1+^q_next/Adam/update_q_next/biases/ApplyAdam/^q_next/Adam/update_q_next/conv1/bias/ApplyAdam1^q_next/Adam/update_q_next/conv1/kernel/ApplyAdam/^q_next/Adam/update_q_next/conv2/bias/ApplyAdam1^q_next/Adam/update_q_next/conv2/kernel/ApplyAdam7^q_next/Adam/update_q_next/rnn/lstm_cell/bias/ApplyAdam9^q_next/Adam/update_q_next/rnn/lstm_cell/kernel/ApplyAdam,^q_next/Adam/update_q_next/weights/ApplyAdam

init_1NoOp^q_eval/beta1_power/Assign^q_eval/beta2_power/Assign^q_eval/biases/Assign^q_eval/conv1/bias/Assign^q_eval/conv1/kernel/Assign^q_eval/conv2/bias/Assign^q_eval/conv2/kernel/Assign!^q_eval/q_eval/biases/Adam/Assign#^q_eval/q_eval/biases/Adam_1/Assign%^q_eval/q_eval/conv1/bias/Adam/Assign'^q_eval/q_eval/conv1/bias/Adam_1/Assign'^q_eval/q_eval/conv1/kernel/Adam/Assign)^q_eval/q_eval/conv1/kernel/Adam_1/Assign%^q_eval/q_eval/conv2/bias/Adam/Assign'^q_eval/q_eval/conv2/bias/Adam_1/Assign'^q_eval/q_eval/conv2/kernel/Adam/Assign)^q_eval/q_eval/conv2/kernel/Adam_1/Assign-^q_eval/q_eval/rnn/lstm_cell/bias/Adam/Assign/^q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Assign/^q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Assign1^q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Assign"^q_eval/q_eval/weights/Adam/Assign$^q_eval/q_eval/weights/Adam_1/Assign!^q_eval/rnn/lstm_cell/bias/Assign#^q_eval/rnn/lstm_cell/kernel/Assign^q_eval/weights/Assign^q_next/beta1_power/Assign^q_next/beta2_power/Assign^q_next/biases/Assign^q_next/conv1/bias/Assign^q_next/conv1/kernel/Assign^q_next/conv2/bias/Assign^q_next/conv2/kernel/Assign!^q_next/q_next/biases/Adam/Assign#^q_next/q_next/biases/Adam_1/Assign%^q_next/q_next/conv1/bias/Adam/Assign'^q_next/q_next/conv1/bias/Adam_1/Assign'^q_next/q_next/conv1/kernel/Adam/Assign)^q_next/q_next/conv1/kernel/Adam_1/Assign%^q_next/q_next/conv2/bias/Adam/Assign'^q_next/q_next/conv2/bias/Adam_1/Assign'^q_next/q_next/conv2/kernel/Adam/Assign)^q_next/q_next/conv2/kernel/Adam_1/Assign-^q_next/q_next/rnn/lstm_cell/bias/Adam/Assign/^q_next/q_next/rnn/lstm_cell/bias/Adam_1/Assign/^q_next/q_next/rnn/lstm_cell/kernel/Adam/Assign1^q_next/q_next/rnn/lstm_cell/kernel/Adam_1/Assign"^q_next/q_next/weights/Adam/Assign$^q_next/q_next/weights/Adam_1/Assign!^q_next/rnn/lstm_cell/bias/Assign#^q_next/rnn/lstm_cell/kernel/Assign^q_next/weights/Assign
R
save_1/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
О
save_1/SaveV2/tensor_namesConst*я
valueхBт4Bq_eval/beta1_powerBq_eval/beta2_powerBq_eval/biasesBq_eval/conv1/biasBq_eval/conv1/kernelBq_eval/conv2/biasBq_eval/conv2/kernelBq_eval/q_eval/biases/AdamBq_eval/q_eval/biases/Adam_1Bq_eval/q_eval/conv1/bias/AdamBq_eval/q_eval/conv1/bias/Adam_1Bq_eval/q_eval/conv1/kernel/AdamB!q_eval/q_eval/conv1/kernel/Adam_1Bq_eval/q_eval/conv2/bias/AdamBq_eval/q_eval/conv2/bias/Adam_1Bq_eval/q_eval/conv2/kernel/AdamB!q_eval/q_eval/conv2/kernel/Adam_1B%q_eval/q_eval/rnn/lstm_cell/bias/AdamB'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1B'q_eval/q_eval/rnn/lstm_cell/kernel/AdamB)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1Bq_eval/q_eval/weights/AdamBq_eval/q_eval/weights/Adam_1Bq_eval/rnn/lstm_cell/biasBq_eval/rnn/lstm_cell/kernelBq_eval/weightsBq_next/beta1_powerBq_next/beta2_powerBq_next/biasesBq_next/conv1/biasBq_next/conv1/kernelBq_next/conv2/biasBq_next/conv2/kernelBq_next/q_next/biases/AdamBq_next/q_next/biases/Adam_1Bq_next/q_next/conv1/bias/AdamBq_next/q_next/conv1/bias/Adam_1Bq_next/q_next/conv1/kernel/AdamB!q_next/q_next/conv1/kernel/Adam_1Bq_next/q_next/conv2/bias/AdamBq_next/q_next/conv2/bias/Adam_1Bq_next/q_next/conv2/kernel/AdamB!q_next/q_next/conv2/kernel/Adam_1B%q_next/q_next/rnn/lstm_cell/bias/AdamB'q_next/q_next/rnn/lstm_cell/bias/Adam_1B'q_next/q_next/rnn/lstm_cell/kernel/AdamB)q_next/q_next/rnn/lstm_cell/kernel/Adam_1Bq_next/q_next/weights/AdamBq_next/q_next/weights/Adam_1Bq_next/rnn/lstm_cell/biasBq_next/rnn/lstm_cell/kernelBq_next/weights*
dtype0*
_output_shapes
:4
Э
save_1/SaveV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
џ
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesq_eval/beta1_powerq_eval/beta2_powerq_eval/biasesq_eval/conv1/biasq_eval/conv1/kernelq_eval/conv2/biasq_eval/conv2/kernelq_eval/q_eval/biases/Adamq_eval/q_eval/biases/Adam_1q_eval/q_eval/conv1/bias/Adamq_eval/q_eval/conv1/bias/Adam_1q_eval/q_eval/conv1/kernel/Adam!q_eval/q_eval/conv1/kernel/Adam_1q_eval/q_eval/conv2/bias/Adamq_eval/q_eval/conv2/bias/Adam_1q_eval/q_eval/conv2/kernel/Adam!q_eval/q_eval/conv2/kernel/Adam_1%q_eval/q_eval/rnn/lstm_cell/bias/Adam'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1'q_eval/q_eval/rnn/lstm_cell/kernel/Adam)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1q_eval/q_eval/weights/Adamq_eval/q_eval/weights/Adam_1q_eval/rnn/lstm_cell/biasq_eval/rnn/lstm_cell/kernelq_eval/weightsq_next/beta1_powerq_next/beta2_powerq_next/biasesq_next/conv1/biasq_next/conv1/kernelq_next/conv2/biasq_next/conv2/kernelq_next/q_next/biases/Adamq_next/q_next/biases/Adam_1q_next/q_next/conv1/bias/Adamq_next/q_next/conv1/bias/Adam_1q_next/q_next/conv1/kernel/Adam!q_next/q_next/conv1/kernel/Adam_1q_next/q_next/conv2/bias/Adamq_next/q_next/conv2/bias/Adam_1q_next/q_next/conv2/kernel/Adam!q_next/q_next/conv2/kernel/Adam_1%q_next/q_next/rnn/lstm_cell/bias/Adam'q_next/q_next/rnn/lstm_cell/bias/Adam_1'q_next/q_next/rnn/lstm_cell/kernel/Adam)q_next/q_next/rnn/lstm_cell/kernel/Adam_1q_next/q_next/weights/Adamq_next/q_next/weights/Adam_1q_next/rnn/lstm_cell/biasq_next/rnn/lstm_cell/kernelq_next/weights*B
dtypes8
624

save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save_1/Const
а
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*я
valueхBт4Bq_eval/beta1_powerBq_eval/beta2_powerBq_eval/biasesBq_eval/conv1/biasBq_eval/conv1/kernelBq_eval/conv2/biasBq_eval/conv2/kernelBq_eval/q_eval/biases/AdamBq_eval/q_eval/biases/Adam_1Bq_eval/q_eval/conv1/bias/AdamBq_eval/q_eval/conv1/bias/Adam_1Bq_eval/q_eval/conv1/kernel/AdamB!q_eval/q_eval/conv1/kernel/Adam_1Bq_eval/q_eval/conv2/bias/AdamBq_eval/q_eval/conv2/bias/Adam_1Bq_eval/q_eval/conv2/kernel/AdamB!q_eval/q_eval/conv2/kernel/Adam_1B%q_eval/q_eval/rnn/lstm_cell/bias/AdamB'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1B'q_eval/q_eval/rnn/lstm_cell/kernel/AdamB)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1Bq_eval/q_eval/weights/AdamBq_eval/q_eval/weights/Adam_1Bq_eval/rnn/lstm_cell/biasBq_eval/rnn/lstm_cell/kernelBq_eval/weightsBq_next/beta1_powerBq_next/beta2_powerBq_next/biasesBq_next/conv1/biasBq_next/conv1/kernelBq_next/conv2/biasBq_next/conv2/kernelBq_next/q_next/biases/AdamBq_next/q_next/biases/Adam_1Bq_next/q_next/conv1/bias/AdamBq_next/q_next/conv1/bias/Adam_1Bq_next/q_next/conv1/kernel/AdamB!q_next/q_next/conv1/kernel/Adam_1Bq_next/q_next/conv2/bias/AdamBq_next/q_next/conv2/bias/Adam_1Bq_next/q_next/conv2/kernel/AdamB!q_next/q_next/conv2/kernel/Adam_1B%q_next/q_next/rnn/lstm_cell/bias/AdamB'q_next/q_next/rnn/lstm_cell/bias/Adam_1B'q_next/q_next/rnn/lstm_cell/kernel/AdamB)q_next/q_next/rnn/lstm_cell/kernel/Adam_1Bq_next/q_next/weights/AdamBq_next/q_next/weights/Adam_1Bq_next/rnn/lstm_cell/biasBq_next/rnn/lstm_cell/kernelBq_next/weights*
dtype0*
_output_shapes
:4
п
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
Љ
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*ц
_output_shapesг
а::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
Љ
save_1/AssignAssignq_eval/beta1_powersave_1/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@q_eval/biases
­
save_1/Assign_1Assignq_eval/beta2_powersave_1/RestoreV2:1*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: *
use_locking(
Ќ
save_1/Assign_2Assignq_eval/biasessave_1/RestoreV2:2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_eval/biases
Д
save_1/Assign_3Assignq_eval/conv1/biassave_1/RestoreV2:3*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:
Ф
save_1/Assign_4Assignq_eval/conv1/kernelsave_1/RestoreV2:4*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:
Д
save_1/Assign_5Assignq_eval/conv2/biassave_1/RestoreV2:5*
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: 
Ф
save_1/Assign_6Assignq_eval/conv2/kernelsave_1/RestoreV2:6*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel
И
save_1/Assign_7Assignq_eval/q_eval/biases/Adamsave_1/RestoreV2:7*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_eval/biases
К
save_1/Assign_8Assignq_eval/q_eval/biases/Adam_1save_1/RestoreV2:8*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:*
use_locking(
Р
save_1/Assign_9Assignq_eval/q_eval/conv1/bias/Adamsave_1/RestoreV2:9*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:
Ф
save_1/Assign_10Assignq_eval/q_eval/conv1/bias/Adam_1save_1/RestoreV2:10*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:
в
save_1/Assign_11Assignq_eval/q_eval/conv1/kernel/Adamsave_1/RestoreV2:11*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:
д
save_1/Assign_12Assign!q_eval/q_eval/conv1/kernel/Adam_1save_1/RestoreV2:12*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel
Т
save_1/Assign_13Assignq_eval/q_eval/conv2/bias/Adamsave_1/RestoreV2:13*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ф
save_1/Assign_14Assignq_eval/q_eval/conv2/bias/Adam_1save_1/RestoreV2:14*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: *
use_locking(
в
save_1/Assign_15Assignq_eval/q_eval/conv2/kernel/Adamsave_1/RestoreV2:15*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(
д
save_1/Assign_16Assign!q_eval/q_eval/conv2/kernel/Adam_1save_1/RestoreV2:16*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(
г
save_1/Assign_17Assign%q_eval/q_eval/rnn/lstm_cell/bias/Adamsave_1/RestoreV2:17*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias
е
save_1/Assign_18Assign'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1save_1/RestoreV2:18*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias
м
save_1/Assign_19Assign'q_eval/q_eval/rnn/lstm_cell/kernel/Adamsave_1/RestoreV2:19*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
о
save_1/Assign_20Assign)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1save_1/RestoreV2:20*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
С
save_1/Assign_21Assignq_eval/q_eval/weights/Adamsave_1/RestoreV2:21*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
У
save_1/Assign_22Assignq_eval/q_eval/weights/Adam_1save_1/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
Ч
save_1/Assign_23Assignq_eval/rnn/lstm_cell/biassave_1/RestoreV2:23*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
а
save_1/Assign_24Assignq_eval/rnn/lstm_cell/kernelsave_1/RestoreV2:24*
use_locking(*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
Е
save_1/Assign_25Assignq_eval/weightssave_1/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
Џ
save_1/Assign_26Assignq_next/beta1_powersave_1/RestoreV2:26*
use_locking(*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: 
Џ
save_1/Assign_27Assignq_next/beta2_powersave_1/RestoreV2:27*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: *
use_locking(
Ў
save_1/Assign_28Assignq_next/biasessave_1/RestoreV2:28*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
:*
use_locking(
Ж
save_1/Assign_29Assignq_next/conv1/biassave_1/RestoreV2:29*
T0*$
_class
loc:@q_next/conv1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ц
save_1/Assign_30Assignq_next/conv1/kernelsave_1/RestoreV2:30*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_next/conv1/kernel
Ж
save_1/Assign_31Assignq_next/conv2/biassave_1/RestoreV2:31*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias
Ц
save_1/Assign_32Assignq_next/conv2/kernelsave_1/RestoreV2:32*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel
К
save_1/Assign_33Assignq_next/q_next/biases/Adamsave_1/RestoreV2:33*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
:*
use_locking(
М
save_1/Assign_34Assignq_next/q_next/biases/Adam_1save_1/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_next/biases
Т
save_1/Assign_35Assignq_next/q_next/conv1/bias/Adamsave_1/RestoreV2:35*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_next/conv1/bias
Ф
save_1/Assign_36Assignq_next/q_next/conv1/bias/Adam_1save_1/RestoreV2:36*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_next/conv1/bias
в
save_1/Assign_37Assignq_next/q_next/conv1/kernel/Adamsave_1/RestoreV2:37*
use_locking(*
T0*&
_class
loc:@q_next/conv1/kernel*
validate_shape(*&
_output_shapes
:
д
save_1/Assign_38Assign!q_next/q_next/conv1/kernel/Adam_1save_1/RestoreV2:38*
T0*&
_class
loc:@q_next/conv1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
Т
save_1/Assign_39Assignq_next/q_next/conv2/bias/Adamsave_1/RestoreV2:39*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias
Ф
save_1/Assign_40Assignq_next/q_next/conv2/bias/Adam_1save_1/RestoreV2:40*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias
в
save_1/Assign_41Assignq_next/q_next/conv2/kernel/Adamsave_1/RestoreV2:41*
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel*
validate_shape(*&
_output_shapes
: 
д
save_1/Assign_42Assign!q_next/q_next/conv2/kernel/Adam_1save_1/RestoreV2:42*
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel*
validate_shape(*&
_output_shapes
: 
г
save_1/Assign_43Assign%q_next/q_next/rnn/lstm_cell/bias/Adamsave_1/RestoreV2:43*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
е
save_1/Assign_44Assign'q_next/q_next/rnn/lstm_cell/bias/Adam_1save_1/RestoreV2:44*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
м
save_1/Assign_45Assign'q_next/q_next/rnn/lstm_cell/kernel/Adamsave_1/RestoreV2:45*
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
о
save_1/Assign_46Assign)q_next/q_next/rnn/lstm_cell/kernel/Adam_1save_1/RestoreV2:46*
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
С
save_1/Assign_47Assignq_next/q_next/weights/Adamsave_1/RestoreV2:47*
use_locking(*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	
У
save_1/Assign_48Assignq_next/q_next/weights/Adam_1save_1/RestoreV2:48*
use_locking(*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	
Ч
save_1/Assign_49Assignq_next/rnn/lstm_cell/biassave_1/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
а
save_1/Assign_50Assignq_next/rnn/lstm_cell/kernelsave_1/RestoreV2:50*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
Е
save_1/Assign_51Assignq_next/weightssave_1/RestoreV2:51*
use_locking(*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	
ъ
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
М
Merge_1/MergeSummaryMergeSummaryq_eval/Reward/Time_step_1#q_eval/TotalWaitingTime/Time_step_1q_eval/TotalDelay/Time_step_1q_eval/Q_valueq_eval/Loss.q_eval/q_eval/conv1/kernel/summaries/histogram,q_eval/q_eval/conv1/bias/summaries/histogram.q_eval/q_eval/conv2/kernel/summaries/histogram,q_eval/q_eval/conv2/bias/summaries/histogram6q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram4q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram)q_eval/q_eval/weights/summaries/histogram(q_eval/q_eval/biases/summaries/histogramq_next/Reward/Time_step_1#q_next/TotalWaitingTime/Time_step_1q_next/TotalDelay/Time_step_1q_next/Q_valueq_next/Loss*
N*
_output_shapes
: "руG(Љ     Кsсг	ёэ5nзAJв
ђAвA
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
ю
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
2
L2Loss
t"T
output"T"
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z

!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
1
Square
x"T
y"T"
Ttype:

2	
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( 
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring 
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 

StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
о
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.8.02v1.8.0-0-g93bc2e2072Нь

q_eval/statesPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџTT*$
shape:џџџџџџџџџTT
v
q_eval/action_takenPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
i
q_eval/q_valuePlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
[
q_eval/sequence_lengthPlaceholder*
dtype0*
_output_shapes
:*
shape:
V
q_eval/batch_sizePlaceholder*
dtype0*
_output_shapes
:*
shape:
v
q_eval/cell_statePlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
s
q_eval/h_statePlaceholder*
shape:џџџџџџџџџ*
dtype0*(
_output_shapes
:џџџџџџџџџ
X
q_eval/Reward/Time_stepPlaceholder*
dtype0*
_output_shapes
: *
shape: 
x
q_eval/Reward/Time_step_1/tagsConst**
value!B Bq_eval/Reward/Time_step_1*
dtype0*
_output_shapes
: 

q_eval/Reward/Time_step_1ScalarSummaryq_eval/Reward/Time_step_1/tagsq_eval/Reward/Time_step*
T0*
_output_shapes
: 
b
!q_eval/TotalWaitingTime/Time_stepPlaceholder*
dtype0*
_output_shapes
: *
shape: 

(q_eval/TotalWaitingTime/Time_step_1/tagsConst*4
value+B) B#q_eval/TotalWaitingTime/Time_step_1*
dtype0*
_output_shapes
: 
Ђ
#q_eval/TotalWaitingTime/Time_step_1ScalarSummary(q_eval/TotalWaitingTime/Time_step_1/tags!q_eval/TotalWaitingTime/Time_step*
T0*
_output_shapes
: 
\
q_eval/TotalDelay/Time_stepPlaceholder*
shape: *
dtype0*
_output_shapes
: 

"q_eval/TotalDelay/Time_step_1/tagsConst*
dtype0*
_output_shapes
: *.
value%B# Bq_eval/TotalDelay/Time_step_1

q_eval/TotalDelay/Time_step_1ScalarSummary"q_eval/TotalDelay/Time_step_1/tagsq_eval/TotalDelay/Time_step*
T0*
_output_shapes
: 
З
6q_eval/conv1/kernel/Initializer/truncated_normal/shapeConst*&
_class
loc:@q_eval/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ђ
5q_eval/conv1/kernel/Initializer/truncated_normal/meanConst*&
_class
loc:@q_eval/conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
7q_eval/conv1/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_eval/conv1/kernel*
valueB
 *аdN>

@q_eval/conv1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6q_eval/conv1/kernel/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:*

seed *
T0*&
_class
loc:@q_eval/conv1/kernel*
seed2 

4q_eval/conv1/kernel/Initializer/truncated_normal/mulMul@q_eval/conv1/kernel/Initializer/truncated_normal/TruncatedNormal7q_eval/conv1/kernel/Initializer/truncated_normal/stddev*
T0*&
_class
loc:@q_eval/conv1/kernel*&
_output_shapes
:
§
0q_eval/conv1/kernel/Initializer/truncated_normalAdd4q_eval/conv1/kernel/Initializer/truncated_normal/mul5q_eval/conv1/kernel/Initializer/truncated_normal/mean*&
_output_shapes
:*
T0*&
_class
loc:@q_eval/conv1/kernel
П
q_eval/conv1/kernel
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *&
_class
loc:@q_eval/conv1/kernel*
	container *
shape:
э
q_eval/conv1/kernel/AssignAssignq_eval/conv1/kernel0q_eval/conv1/kernel/Initializer/truncated_normal*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:

q_eval/conv1/kernel/readIdentityq_eval/conv1/kernel*&
_output_shapes
:*
T0*&
_class
loc:@q_eval/conv1/kernel

#q_eval/conv1/bias/Initializer/ConstConst*$
_class
loc:@q_eval/conv1/bias*
valueB*
з#<*
dtype0*
_output_shapes
:
Ѓ
q_eval/conv1/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@q_eval/conv1/bias*
	container *
shape:
Ю
q_eval/conv1/bias/AssignAssignq_eval/conv1/bias#q_eval/conv1/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias

q_eval/conv1/bias/readIdentityq_eval/conv1/bias*
T0*$
_class
loc:@q_eval/conv1/bias*
_output_shapes
:
k
q_eval/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
я
q_eval/conv1/Conv2DConv2Dq_eval/statesq_eval/conv1/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ

q_eval/conv1/BiasAddBiasAddq_eval/conv1/Conv2Dq_eval/conv1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ
c
q_eval/ReluReluq_eval/conv1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
З
6q_eval/conv2/kernel/Initializer/truncated_normal/shapeConst*&
_class
loc:@q_eval/conv2/kernel*%
valueB"             *
dtype0*
_output_shapes
:
Ђ
5q_eval/conv2/kernel/Initializer/truncated_normal/meanConst*&
_class
loc:@q_eval/conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
7q_eval/conv2/kernel/Initializer/truncated_normal/stddevConst*&
_class
loc:@q_eval/conv2/kernel*
valueB
 *аdЮ=*
dtype0*
_output_shapes
: 

@q_eval/conv2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6q_eval/conv2/kernel/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
: *

seed *
T0*&
_class
loc:@q_eval/conv2/kernel*
seed2 

4q_eval/conv2/kernel/Initializer/truncated_normal/mulMul@q_eval/conv2/kernel/Initializer/truncated_normal/TruncatedNormal7q_eval/conv2/kernel/Initializer/truncated_normal/stddev*
T0*&
_class
loc:@q_eval/conv2/kernel*&
_output_shapes
: 
§
0q_eval/conv2/kernel/Initializer/truncated_normalAdd4q_eval/conv2/kernel/Initializer/truncated_normal/mul5q_eval/conv2/kernel/Initializer/truncated_normal/mean*
T0*&
_class
loc:@q_eval/conv2/kernel*&
_output_shapes
: 
П
q_eval/conv2/kernel
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_eval/conv2/kernel*
	container *
shape: 
э
q_eval/conv2/kernel/AssignAssignq_eval/conv2/kernel0q_eval/conv2/kernel/Initializer/truncated_normal*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel

q_eval/conv2/kernel/readIdentityq_eval/conv2/kernel*
T0*&
_class
loc:@q_eval/conv2/kernel*&
_output_shapes
: 

#q_eval/conv2/bias/Initializer/ConstConst*$
_class
loc:@q_eval/conv2/bias*
valueB *
з#<*
dtype0*
_output_shapes
: 
Ѓ
q_eval/conv2/bias
VariableV2*$
_class
loc:@q_eval/conv2/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Ю
q_eval/conv2/bias/AssignAssignq_eval/conv2/bias#q_eval/conv2/bias/Initializer/Const*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: *
use_locking(

q_eval/conv2/bias/readIdentityq_eval/conv2/bias*
T0*$
_class
loc:@q_eval/conv2/bias*
_output_shapes
: 
k
q_eval/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
э
q_eval/conv2/Conv2DConv2Dq_eval/Reluq_eval/conv2/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ		 *
	dilations


q_eval/conv2/BiasAddBiasAddq_eval/conv2/Conv2Dq_eval/conv2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ		 
e
q_eval/Relu_1Reluq_eval/conv2/BiasAdd*/
_output_shapes
:џџџџџџџџџ		 *
T0
e
q_eval/Reshape/shapeConst*
valueB"џџџџ 
  *
dtype0*
_output_shapes
:

q_eval/ReshapeReshapeq_eval/Relu_1q_eval/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ 
[
q_eval/Reshape_1/shape/2Const*
value
B : *
dtype0*
_output_shapes
: 

q_eval/Reshape_1/shapePackq_eval/batch_sizeq_eval/sequence_lengthq_eval/Reshape_1/shape/2*
T0*

axis *
N*
_output_shapes
:

q_eval/Reshape_1Reshapeq_eval/Reshapeq_eval/Reshape_1/shape*
T0*
Tshape0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ 
Q
q_eval/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
X
q_eval/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
X
q_eval/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_eval/rnn/rangeRangeq_eval/rnn/range/startq_eval/rnn/Rankq_eval/rnn/range/delta*
_output_shapes
:*

Tidx0
k
q_eval/rnn/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
X
q_eval/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

q_eval/rnn/concatConcatV2q_eval/rnn/concat/values_0q_eval/rnn/rangeq_eval/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

q_eval/rnn/transpose	Transposeq_eval/Reshape_1q_eval/rnn/concat*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
Tperm0*
T0
a
q_eval/rnn/sequence_lengthIdentityq_eval/sequence_length*
T0*
_output_shapes
:
d
q_eval/rnn/ShapeShapeq_eval/rnn/transpose*
T0*
out_type0*
_output_shapes
:
h
q_eval/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
j
 q_eval/rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
j
 q_eval/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
А
q_eval/rnn/strided_sliceStridedSliceq_eval/rnn/Shapeq_eval/rnn/strided_slice/stack q_eval/rnn/strided_slice/stack_1 q_eval/rnn/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
u
q_eval/rnn/Shape_1Shapeq_eval/rnn/sequence_length*#
_output_shapes
:џџџџџџџџџ*
T0*
out_type0
l
q_eval/rnn/stackPackq_eval/rnn/strided_slice*
N*
_output_shapes
:*
T0*

axis 
m
q_eval/rnn/EqualEqualq_eval/rnn/Shape_1q_eval/rnn/stack*
T0*#
_output_shapes
:џџџџџџџџџ
Z
q_eval/rnn/ConstConst*
valueB: *
dtype0*
_output_shapes
:
n
q_eval/rnn/AllAllq_eval/rnn/Equalq_eval/rnn/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 

q_eval/rnn/Assert/ConstConst*
dtype0*
_output_shapes
: *K
valueBB@ B:Expected shape for Tensor q_eval/rnn/sequence_length:0 is 
j
q_eval/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

q_eval/rnn/Assert/Assert/data_0Const*K
valueBB@ B:Expected shape for Tensor q_eval/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
p
q_eval/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
К
q_eval/rnn/Assert/AssertAssertq_eval/rnn/Allq_eval/rnn/Assert/Assert/data_0q_eval/rnn/stackq_eval/rnn/Assert/Assert/data_2q_eval/rnn/Shape_1*
T
2*
	summarize
|
q_eval/rnn/CheckSeqLenIdentityq_eval/rnn/sequence_length^q_eval/rnn/Assert/Assert*
T0*
_output_shapes
:
f
q_eval/rnn/Shape_2Shapeq_eval/rnn/transpose*
T0*
out_type0*
_output_shapes
:
j
 q_eval/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
l
"q_eval/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"q_eval/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
К
q_eval/rnn/strided_slice_1StridedSliceq_eval/rnn/Shape_2 q_eval/rnn/strided_slice_1/stack"q_eval/rnn/strided_slice_1/stack_1"q_eval/rnn/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
q_eval/rnn/Shape_3Shapeq_eval/rnn/transpose*
T0*
out_type0*
_output_shapes
:
j
 q_eval/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
l
"q_eval/rnn/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
l
"q_eval/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
К
q_eval/rnn/strided_slice_2StridedSliceq_eval/rnn/Shape_3 q_eval/rnn/strided_slice_2/stack"q_eval/rnn/strided_slice_2/stack_1"q_eval/rnn/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
[
q_eval/rnn/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 

q_eval/rnn/ExpandDims
ExpandDimsq_eval/rnn/strided_slice_2q_eval/rnn/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
]
q_eval/rnn/Const_1Const*
dtype0*
_output_shapes
:*
valueB:
Z
q_eval/rnn/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 

q_eval/rnn/concat_1ConcatV2q_eval/rnn/ExpandDimsq_eval/rnn/Const_1q_eval/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
[
q_eval/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

q_eval/rnn/zerosFillq_eval/rnn/concat_1q_eval/rnn/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
R
q_eval/rnn/Rank_1Rankq_eval/rnn/CheckSeqLen*
T0*
_output_shapes
: 
Z
q_eval/rnn/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
Z
q_eval/rnn/range_1/deltaConst*
dtype0*
_output_shapes
: *
value	B :

q_eval/rnn/range_1Rangeq_eval/rnn/range_1/startq_eval/rnn/Rank_1q_eval/rnn/range_1/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ

q_eval/rnn/MinMinq_eval/rnn/CheckSeqLenq_eval/rnn/range_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
q_eval/rnn/Rank_2Rankq_eval/rnn/CheckSeqLen*
_output_shapes
: *
T0
Z
q_eval/rnn/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
Z
q_eval/rnn/range_2/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_eval/rnn/range_2Rangeq_eval/rnn/range_2/startq_eval/rnn/Rank_2q_eval/rnn/range_2/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0

q_eval/rnn/MaxMaxq_eval/rnn/CheckSeqLenq_eval/rnn/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Q
q_eval/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 

q_eval/rnn/TensorArrayTensorArrayV3q_eval/rnn/strided_slice_1*6
tensor_array_name!q_eval/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *%
element_shape:џџџџџџџџџ*
clear_after_read(*
dynamic_size( *
identical_element_shapes(

q_eval/rnn/TensorArray_1TensorArrayV3q_eval/rnn/strided_slice_1*5
tensor_array_name q_eval/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *%
element_shape:џџџџџџџџџ *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
w
#q_eval/rnn/TensorArrayUnstack/ShapeShapeq_eval/rnn/transpose*
T0*
out_type0*
_output_shapes
:
{
1q_eval/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
}
3q_eval/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
}
3q_eval/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

+q_eval/rnn/TensorArrayUnstack/strided_sliceStridedSlice#q_eval/rnn/TensorArrayUnstack/Shape1q_eval/rnn/TensorArrayUnstack/strided_slice/stack3q_eval/rnn/TensorArrayUnstack/strided_slice/stack_13q_eval/rnn/TensorArrayUnstack/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
k
)q_eval/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
k
)q_eval/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
р
#q_eval/rnn/TensorArrayUnstack/rangeRange)q_eval/rnn/TensorArrayUnstack/range/start+q_eval/rnn/TensorArrayUnstack/strided_slice)q_eval/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0

Eq_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3q_eval/rnn/TensorArray_1#q_eval/rnn/TensorArrayUnstack/rangeq_eval/rnn/transposeq_eval/rnn/TensorArray_1:1*
T0*'
_class
loc:@q_eval/rnn/transpose*
_output_shapes
: 
V
q_eval/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
d
q_eval/rnn/MaximumMaximumq_eval/rnn/Maximum/xq_eval/rnn/Max*
T0*
_output_shapes
: 
n
q_eval/rnn/MinimumMinimumq_eval/rnn/strided_slice_1q_eval/rnn/Maximum*
T0*
_output_shapes
: 
d
"q_eval/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Т
q_eval/rnn/while/EnterEnter"q_eval/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
Б
q_eval/rnn/while/Enter_1Enterq_eval/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
К
q_eval/rnn/while/Enter_2Enterq_eval/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
Х
q_eval/rnn/while/Enter_3Enterq_eval/cell_state*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*.

frame_name q_eval/rnn/while/while_context
Т
q_eval/rnn/while/Enter_4Enterq_eval/h_state*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*.

frame_name q_eval/rnn/while/while_context

q_eval/rnn/while/MergeMergeq_eval/rnn/while/Enterq_eval/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0

q_eval/rnn/while/Merge_1Mergeq_eval/rnn/while/Enter_1 q_eval/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 

q_eval/rnn/while/Merge_2Mergeq_eval/rnn/while/Enter_2 q_eval/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

q_eval/rnn/while/Merge_3Mergeq_eval/rnn/while/Enter_3 q_eval/rnn/while/NextIteration_3*
T0*
N**
_output_shapes
:џџџџџџџџџ: 

q_eval/rnn/while/Merge_4Mergeq_eval/rnn/while/Enter_4 q_eval/rnn/while/NextIteration_4*
N**
_output_shapes
:џџџџџџџџџ: *
T0
s
q_eval/rnn/while/LessLessq_eval/rnn/while/Mergeq_eval/rnn/while/Less/Enter*
_output_shapes
: *
T0
П
q_eval/rnn/while/Less/EnterEnterq_eval/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
y
q_eval/rnn/while/Less_1Lessq_eval/rnn/while/Merge_1q_eval/rnn/while/Less_1/Enter*
_output_shapes
: *
T0
Й
q_eval/rnn/while/Less_1/EnterEnterq_eval/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context
q
q_eval/rnn/while/LogicalAnd
LogicalAndq_eval/rnn/while/Lessq_eval/rnn/while/Less_1*
_output_shapes
: 
Z
q_eval/rnn/while/LoopCondLoopCondq_eval/rnn/while/LogicalAnd*
_output_shapes
: 
Ђ
q_eval/rnn/while/SwitchSwitchq_eval/rnn/while/Mergeq_eval/rnn/while/LoopCond*
T0*)
_class
loc:@q_eval/rnn/while/Merge*
_output_shapes
: : 
Ј
q_eval/rnn/while/Switch_1Switchq_eval/rnn/while/Merge_1q_eval/rnn/while/LoopCond*
T0*+
_class!
loc:@q_eval/rnn/while/Merge_1*
_output_shapes
: : 
Ј
q_eval/rnn/while/Switch_2Switchq_eval/rnn/while/Merge_2q_eval/rnn/while/LoopCond*
T0*+
_class!
loc:@q_eval/rnn/while/Merge_2*
_output_shapes
: : 
Ь
q_eval/rnn/while/Switch_3Switchq_eval/rnn/while/Merge_3q_eval/rnn/while/LoopCond*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*+
_class!
loc:@q_eval/rnn/while/Merge_3
Ь
q_eval/rnn/while/Switch_4Switchq_eval/rnn/while/Merge_4q_eval/rnn/while/LoopCond*
T0*+
_class!
loc:@q_eval/rnn/while/Merge_4*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
a
q_eval/rnn/while/IdentityIdentityq_eval/rnn/while/Switch:1*
T0*
_output_shapes
: 
e
q_eval/rnn/while/Identity_1Identityq_eval/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
e
q_eval/rnn/while/Identity_2Identityq_eval/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
w
q_eval/rnn/while/Identity_3Identityq_eval/rnn/while/Switch_3:1*
T0*(
_output_shapes
:џџџџџџџџџ
w
q_eval/rnn/while/Identity_4Identityq_eval/rnn/while/Switch_4:1*
T0*(
_output_shapes
:џџџџџџџџџ
t
q_eval/rnn/while/add/yConst^q_eval/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
q_eval/rnn/while/addAddq_eval/rnn/while/Identityq_eval/rnn/while/add/y*
T0*
_output_shapes
: 
с
"q_eval/rnn/while/TensorArrayReadV3TensorArrayReadV3(q_eval/rnn/while/TensorArrayReadV3/Enterq_eval/rnn/while/Identity_1*q_eval/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:џџџџџџџџџ 
Ю
(q_eval/rnn/while/TensorArrayReadV3/EnterEnterq_eval/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
љ
*q_eval/rnn/while/TensorArrayReadV3/Enter_1EnterEq_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context

q_eval/rnn/while/GreaterEqualGreaterEqualq_eval/rnn/while/Identity_1#q_eval/rnn/while/GreaterEqual/Enter*
_output_shapes
:*
T0
Х
#q_eval/rnn/while/GreaterEqual/EnterEnterq_eval/rnn/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Н
<q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB"      
Џ
:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB
 *њ<!Н*
dtype0*
_output_shapes
: 
Џ
:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB
 *њ<!=*
dtype0*
_output_shapes
: 

Dq_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform<q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
seed2 *
dtype0* 
_output_shapes
:
 *

seed 

:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/subSub:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/max:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
_output_shapes
: 

:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMulDq_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniform:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel* 
_output_shapes
:
 

6q_eval/rnn/lstm_cell/kernel/Initializer/random_uniformAdd:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/mul:q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel* 
_output_shapes
:
 
У
q_eval/rnn/lstm_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
 *
shared_name *.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
	container *
shape:
 

"q_eval/rnn/lstm_cell/kernel/AssignAssignq_eval/rnn/lstm_cell/kernel6q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
t
 q_eval/rnn/lstm_cell/kernel/readIdentityq_eval/rnn/lstm_cell/kernel*
T0* 
_output_shapes
:
 
Д
;q_eval/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
Є
1q_eval/rnn/lstm_cell/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB
 *    

+q_eval/rnn/lstm_cell/bias/Initializer/zerosFill;q_eval/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensor1q_eval/rnn/lstm_cell/bias/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
Е
q_eval/rnn/lstm_cell/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
	container 
я
 q_eval/rnn/lstm_cell/bias/AssignAssignq_eval/rnn/lstm_cell/bias+q_eval/rnn/lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
k
q_eval/rnn/lstm_cell/bias/readIdentityq_eval/rnn/lstm_cell/bias*
T0*
_output_shapes	
:

&q_eval/rnn/while/lstm_cell/concat/axisConst^q_eval/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
о
!q_eval/rnn/while/lstm_cell/concatConcatV2"q_eval/rnn/while/TensorArrayReadV3q_eval/rnn/while/Identity_4&q_eval/rnn/while/lstm_cell/concat/axis*
T0*
N*(
_output_shapes
:џџџџџџџџџ *

Tidx0
а
!q_eval/rnn/while/lstm_cell/MatMulMatMul!q_eval/rnn/while/lstm_cell/concat'q_eval/rnn/while/lstm_cell/MatMul/Enter*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
л
'q_eval/rnn/while/lstm_cell/MatMul/EnterEnter q_eval/rnn/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
 *.

frame_name q_eval/rnn/while/while_context
Ф
"q_eval/rnn/while/lstm_cell/BiasAddBiasAdd!q_eval/rnn/while/lstm_cell/MatMul(q_eval/rnn/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
е
(q_eval/rnn/while/lstm_cell/BiasAdd/EnterEnterq_eval/rnn/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*.

frame_name q_eval/rnn/while/while_context
~
 q_eval/rnn/while/lstm_cell/ConstConst^q_eval/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

*q_eval/rnn/while/lstm_cell/split/split_dimConst^q_eval/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
љ
 q_eval/rnn/while/lstm_cell/splitSplit*q_eval/rnn/while/lstm_cell/split/split_dim"q_eval/rnn/while/lstm_cell/BiasAdd*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split

 q_eval/rnn/while/lstm_cell/add/yConst^q_eval/rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

q_eval/rnn/while/lstm_cell/addAdd"q_eval/rnn/while/lstm_cell/split:2 q_eval/rnn/while/lstm_cell/add/y*
T0*(
_output_shapes
:џџџџџџџџџ

"q_eval/rnn/while/lstm_cell/SigmoidSigmoidq_eval/rnn/while/lstm_cell/add*(
_output_shapes
:џџџџџџџџџ*
T0

q_eval/rnn/while/lstm_cell/mulMul"q_eval/rnn/while/lstm_cell/Sigmoidq_eval/rnn/while/Identity_3*
T0*(
_output_shapes
:џџџџџџџџџ

$q_eval/rnn/while/lstm_cell/Sigmoid_1Sigmoid q_eval/rnn/while/lstm_cell/split*(
_output_shapes
:џџџџџџџџџ*
T0
~
q_eval/rnn/while/lstm_cell/TanhTanh"q_eval/rnn/while/lstm_cell/split:1*
T0*(
_output_shapes
:џџџџџџџџџ
Ё
 q_eval/rnn/while/lstm_cell/mul_1Mul$q_eval/rnn/while/lstm_cell/Sigmoid_1q_eval/rnn/while/lstm_cell/Tanh*
T0*(
_output_shapes
:џџџџџџџџџ

 q_eval/rnn/while/lstm_cell/add_1Addq_eval/rnn/while/lstm_cell/mul q_eval/rnn/while/lstm_cell/mul_1*(
_output_shapes
:џџџџџџџџџ*
T0

$q_eval/rnn/while/lstm_cell/Sigmoid_2Sigmoid"q_eval/rnn/while/lstm_cell/split:3*
T0*(
_output_shapes
:џџџџџџџџџ
~
!q_eval/rnn/while/lstm_cell/Tanh_1Tanh q_eval/rnn/while/lstm_cell/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
Ѓ
 q_eval/rnn/while/lstm_cell/mul_2Mul$q_eval/rnn/while/lstm_cell/Sigmoid_2!q_eval/rnn/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:џџџџџџџџџ
щ
q_eval/rnn/while/SelectSelectq_eval/rnn/while/GreaterEqualq_eval/rnn/while/Select/Enter q_eval/rnn/while/lstm_cell/mul_2*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*(
_output_shapes
:џџџџџџџџџ
ў
q_eval/rnn/while/Select/EnterEnterq_eval/rnn/zeros*
is_constant(*(
_output_shapes
:џџџџџџџџџ*.

frame_name q_eval/rnn/while/while_context*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
parallel_iterations 
щ
q_eval/rnn/while/Select_1Selectq_eval/rnn/while/GreaterEqualq_eval/rnn/while/Identity_3 q_eval/rnn/while/lstm_cell/add_1*(
_output_shapes
:џџџџџџџџџ*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/add_1
щ
q_eval/rnn/while/Select_2Selectq_eval/rnn/while/GreaterEqualq_eval/rnn/while/Identity_4 q_eval/rnn/while/lstm_cell/mul_2*(
_output_shapes
:џџџџџџџџџ*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2
Џ
4q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3:q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterq_eval/rnn/while/Identity_1q_eval/rnn/while/Selectq_eval/rnn/while/Identity_2*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
_output_shapes
: 

:q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterq_eval/rnn/TensorArray*
is_constant(*
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
parallel_iterations 
v
q_eval/rnn/while/add_1/yConst^q_eval/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
q_eval/rnn/while/add_1Addq_eval/rnn/while/Identity_1q_eval/rnn/while/add_1/y*
T0*
_output_shapes
: 
f
q_eval/rnn/while/NextIterationNextIterationq_eval/rnn/while/add*
T0*
_output_shapes
: 
j
 q_eval/rnn/while/NextIteration_1NextIterationq_eval/rnn/while/add_1*
T0*
_output_shapes
: 

 q_eval/rnn/while/NextIteration_2NextIteration4q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

 q_eval/rnn/while/NextIteration_3NextIterationq_eval/rnn/while/Select_1*
T0*(
_output_shapes
:џџџџџџџџџ

 q_eval/rnn/while/NextIteration_4NextIterationq_eval/rnn/while/Select_2*(
_output_shapes
:џџџџџџџџџ*
T0
W
q_eval/rnn/while/ExitExitq_eval/rnn/while/Switch*
T0*
_output_shapes
: 
[
q_eval/rnn/while/Exit_1Exitq_eval/rnn/while/Switch_1*
T0*
_output_shapes
: 
[
q_eval/rnn/while/Exit_2Exitq_eval/rnn/while/Switch_2*
T0*
_output_shapes
: 
m
q_eval/rnn/while/Exit_3Exitq_eval/rnn/while/Switch_3*(
_output_shapes
:џџџџџџџџџ*
T0
m
q_eval/rnn/while/Exit_4Exitq_eval/rnn/while/Switch_4*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
-q_eval/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3q_eval/rnn/TensorArrayq_eval/rnn/while/Exit_2*)
_class
loc:@q_eval/rnn/TensorArray*
_output_shapes
: 

'q_eval/rnn/TensorArrayStack/range/startConst*)
_class
loc:@q_eval/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

'q_eval/rnn/TensorArrayStack/range/deltaConst*)
_class
loc:@q_eval/rnn/TensorArray*
value	B :*
dtype0*
_output_shapes
: 

!q_eval/rnn/TensorArrayStack/rangeRange'q_eval/rnn/TensorArrayStack/range/start-q_eval/rnn/TensorArrayStack/TensorArraySizeV3'q_eval/rnn/TensorArrayStack/range/delta*)
_class
loc:@q_eval/rnn/TensorArray*#
_output_shapes
:џџџџџџџџџ*

Tidx0
А
/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3q_eval/rnn/TensorArray!q_eval/rnn/TensorArrayStack/rangeq_eval/rnn/while/Exit_2*%
element_shape:џџџџџџџџџ*)
_class
loc:@q_eval/rnn/TensorArray*
dtype0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
]
q_eval/rnn/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
S
q_eval/rnn/Rank_3Const*
value	B :*
dtype0*
_output_shapes
: 
Z
q_eval/rnn/range_3/startConst*
value	B :*
dtype0*
_output_shapes
: 
Z
q_eval/rnn/range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_eval/rnn/range_3Rangeq_eval/rnn/range_3/startq_eval/rnn/Rank_3q_eval/rnn/range_3/delta*
_output_shapes
:*

Tidx0
m
q_eval/rnn/concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
Z
q_eval/rnn/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ѕ
q_eval/rnn/concat_2ConcatV2q_eval/rnn/concat_2/values_0q_eval/rnn/range_3q_eval/rnn/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ж
q_eval/rnn/transpose_1	Transpose/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3q_eval/rnn/concat_2*
Tperm0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
Ѓ
/q_eval/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@q_eval/weights*
valueB"      

-q_eval/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *!
_class
loc:@q_eval/weights*
valueB
 *О

-q_eval/weights/Initializer/random_uniform/maxConst*!
_class
loc:@q_eval/weights*
valueB
 *>*
dtype0*
_output_shapes
: 
ь
7q_eval/weights/Initializer/random_uniform/RandomUniformRandomUniform/q_eval/weights/Initializer/random_uniform/shape*

seed *
T0*!
_class
loc:@q_eval/weights*
seed2 *
dtype0*
_output_shapes
:	
ж
-q_eval/weights/Initializer/random_uniform/subSub-q_eval/weights/Initializer/random_uniform/max-q_eval/weights/Initializer/random_uniform/min*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
: 
щ
-q_eval/weights/Initializer/random_uniform/mulMul7q_eval/weights/Initializer/random_uniform/RandomUniform-q_eval/weights/Initializer/random_uniform/sub*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
:	
л
)q_eval/weights/Initializer/random_uniformAdd-q_eval/weights/Initializer/random_uniform/mul-q_eval/weights/Initializer/random_uniform/min*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
:	
Ї
q_eval/weights
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *!
_class
loc:@q_eval/weights*
	container *
shape:	
а
q_eval/weights/AssignAssignq_eval/weights)q_eval/weights/Initializer/random_uniform*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	*
use_locking(
|
q_eval/weights/readIdentityq_eval/weights*
_output_shapes
:	*
T0*!
_class
loc:@q_eval/weights

/q_eval/weights/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *!
_class
loc:@q_eval/weights*
valueB
 *
з#<

0q_eval/weights/Regularizer/l2_regularizer/L2LossL2Lossq_eval/weights/read*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
: 
з
)q_eval/weights/Regularizer/l2_regularizerMul/q_eval/weights/Regularizer/l2_regularizer/scale0q_eval/weights/Regularizer/l2_regularizer/L2Loss*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
: 

q_eval/biases/Initializer/ConstConst* 
_class
loc:@q_eval/biases*
valueB*
з#<*
dtype0*
_output_shapes
:

q_eval/biases
VariableV2*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@q_eval/biases*
	container *
shape:
О
q_eval/biases/AssignAssignq_eval/biasesq_eval/biases/Initializer/Const*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:*
use_locking(
t
q_eval/biases/readIdentityq_eval/biases*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
:
o
q_eval/strided_slice/stackConst*!
valueB"    џџџџ    *
dtype0*
_output_shapes
:
q
q_eval/strided_slice/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
q
q_eval/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
И
q_eval/strided_sliceStridedSliceq_eval/rnn/transpose_1q_eval/strided_slice/stackq_eval/strided_slice/stack_1q_eval/strided_slice/stack_2*
end_mask*(
_output_shapes
:џџџџџџџџџ*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask 

q_eval/MatMulMatMulq_eval/strided_sliceq_eval/weights/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
f

q_eval/addAddq_eval/MatMulq_eval/biases/read*
T0*'
_output_shapes
:џџџџџџџџџ
a
q_eval/Q_value/tagConst*
valueB Bq_eval/Q_value*
dtype0*
_output_shapes
: 
c
q_eval/Q_valueHistogramSummaryq_eval/Q_value/tag
q_eval/add*
T0*
_output_shapes
: 
d

q_eval/MulMul
q_eval/addq_eval/action_taken*'
_output_shapes
:џџџџџџџџџ*
T0
^
q_eval/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 


q_eval/SumSum
q_eval/Mulq_eval/Sum/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0*
T0
[

q_eval/subSubq_eval/q_value
q_eval/Sum*
T0*#
_output_shapes
:џџџџџџџџџ
Q
q_eval/SquareSquare
q_eval/sub*
T0*#
_output_shapes
:џџџџџџџџџ
V
q_eval/ConstConst*
valueB: *
dtype0*
_output_shapes
:
n
q_eval/MeanMeanq_eval/Squareq_eval/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
\
q_eval/Loss/tagsConst*
dtype0*
_output_shapes
: *
valueB Bq_eval/Loss
\
q_eval/LossScalarSummaryq_eval/Loss/tagsq_eval/Mean*
T0*
_output_shapes
: 
Y
q_eval/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
q_eval/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

q_eval/gradients/FillFillq_eval/gradients/Shapeq_eval/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
Z
q_eval/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
М
q_eval/gradients/f_count_1Enterq_eval/gradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_eval/rnn/while/while_context

q_eval/gradients/MergeMergeq_eval/gradients/f_count_1q_eval/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
w
q_eval/gradients/SwitchSwitchq_eval/gradients/Mergeq_eval/rnn/while/LoopCond*
T0*
_output_shapes
: : 
t
q_eval/gradients/Add/yConst^q_eval/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
q_eval/gradients/AddAddq_eval/gradients/Switch:1q_eval/gradients/Add/y*
_output_shapes
: *
T0
ъ
q_eval/gradients/NextIterationNextIterationq_eval/gradients/AddC^q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPushV2G^q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPushV2G^q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPushV2i^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2M^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2Y^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2[^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1W^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2K^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2Y^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2[^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1G^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2I^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2Y^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2[^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1G^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2I^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2W^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2Y^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1G^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
_output_shapes
: *
T0
\
q_eval/gradients/f_count_2Exitq_eval/gradients/Switch*
T0*
_output_shapes
: 
Z
q_eval/gradients/b_countConst*
dtype0*
_output_shapes
: *
value	B :
Я
q_eval/gradients/b_count_1Enterq_eval/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

q_eval/gradients/Merge_1Mergeq_eval/gradients/b_count_1 q_eval/gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 

q_eval/gradients/GreaterEqualGreaterEqualq_eval/gradients/Merge_1#q_eval/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
ж
#q_eval/gradients/GreaterEqual/EnterEnterq_eval/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
]
q_eval/gradients/b_count_2LoopCondq_eval/gradients/GreaterEqual*
_output_shapes
: 
|
q_eval/gradients/Switch_1Switchq_eval/gradients/Merge_1q_eval/gradients/b_count_2*
T0*
_output_shapes
: : 
~
q_eval/gradients/SubSubq_eval/gradients/Switch_1:1#q_eval/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
Ю
 q_eval/gradients/NextIteration_1NextIterationq_eval/gradients/Subd^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
^
q_eval/gradients/b_count_3Exitq_eval/gradients/Switch_1*
T0*
_output_shapes
: 
y
/q_eval/gradients/q_eval/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Џ
)q_eval/gradients/q_eval/Mean_grad/ReshapeReshapeq_eval/gradients/Fill/q_eval/gradients/q_eval/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
t
'q_eval/gradients/q_eval/Mean_grad/ShapeShapeq_eval/Square*
T0*
out_type0*
_output_shapes
:
Т
&q_eval/gradients/q_eval/Mean_grad/TileTile)q_eval/gradients/q_eval/Mean_grad/Reshape'q_eval/gradients/q_eval/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
v
)q_eval/gradients/q_eval/Mean_grad/Shape_1Shapeq_eval/Square*
T0*
out_type0*
_output_shapes
:
l
)q_eval/gradients/q_eval/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
q
'q_eval/gradients/q_eval/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Р
&q_eval/gradients/q_eval/Mean_grad/ProdProd)q_eval/gradients/q_eval/Mean_grad/Shape_1'q_eval/gradients/q_eval/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
s
)q_eval/gradients/q_eval/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ф
(q_eval/gradients/q_eval/Mean_grad/Prod_1Prod)q_eval/gradients/q_eval/Mean_grad/Shape_2)q_eval/gradients/q_eval/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
m
+q_eval/gradients/q_eval/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ќ
)q_eval/gradients/q_eval/Mean_grad/MaximumMaximum(q_eval/gradients/q_eval/Mean_grad/Prod_1+q_eval/gradients/q_eval/Mean_grad/Maximum/y*
_output_shapes
: *
T0
Њ
*q_eval/gradients/q_eval/Mean_grad/floordivFloorDiv&q_eval/gradients/q_eval/Mean_grad/Prod)q_eval/gradients/q_eval/Mean_grad/Maximum*
T0*
_output_shapes
: 

&q_eval/gradients/q_eval/Mean_grad/CastCast*q_eval/gradients/q_eval/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
В
)q_eval/gradients/q_eval/Mean_grad/truedivRealDiv&q_eval/gradients/q_eval/Mean_grad/Tile&q_eval/gradients/q_eval/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ

)q_eval/gradients/q_eval/Square_grad/ConstConst*^q_eval/gradients/q_eval/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 

'q_eval/gradients/q_eval/Square_grad/MulMul
q_eval/sub)q_eval/gradients/q_eval/Square_grad/Const*#
_output_shapes
:џџџџџџџџџ*
T0
В
)q_eval/gradients/q_eval/Square_grad/Mul_1Mul)q_eval/gradients/q_eval/Mean_grad/truediv'q_eval/gradients/q_eval/Square_grad/Mul*#
_output_shapes
:џџџџџџџџџ*
T0
t
&q_eval/gradients/q_eval/sub_grad/ShapeShapeq_eval/q_value*
T0*
out_type0*
_output_shapes
:
r
(q_eval/gradients/q_eval/sub_grad/Shape_1Shape
q_eval/Sum*
_output_shapes
:*
T0*
out_type0
о
6q_eval/gradients/q_eval/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&q_eval/gradients/q_eval/sub_grad/Shape(q_eval/gradients/q_eval/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ю
$q_eval/gradients/q_eval/sub_grad/SumSum)q_eval/gradients/q_eval/Square_grad/Mul_16q_eval/gradients/q_eval/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
(q_eval/gradients/q_eval/sub_grad/ReshapeReshape$q_eval/gradients/q_eval/sub_grad/Sum&q_eval/gradients/q_eval/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
в
&q_eval/gradients/q_eval/sub_grad/Sum_1Sum)q_eval/gradients/q_eval/Square_grad/Mul_18q_eval/gradients/q_eval/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
v
$q_eval/gradients/q_eval/sub_grad/NegNeg&q_eval/gradients/q_eval/sub_grad/Sum_1*
_output_shapes
:*
T0
С
*q_eval/gradients/q_eval/sub_grad/Reshape_1Reshape$q_eval/gradients/q_eval/sub_grad/Neg(q_eval/gradients/q_eval/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

1q_eval/gradients/q_eval/sub_grad/tuple/group_depsNoOp)^q_eval/gradients/q_eval/sub_grad/Reshape+^q_eval/gradients/q_eval/sub_grad/Reshape_1

9q_eval/gradients/q_eval/sub_grad/tuple/control_dependencyIdentity(q_eval/gradients/q_eval/sub_grad/Reshape2^q_eval/gradients/q_eval/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_eval/gradients/q_eval/sub_grad/Reshape*#
_output_shapes
:џџџџџџџџџ

;q_eval/gradients/q_eval/sub_grad/tuple/control_dependency_1Identity*q_eval/gradients/q_eval/sub_grad/Reshape_12^q_eval/gradients/q_eval/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/sub_grad/Reshape_1*#
_output_shapes
:џџџџџџџџџ
p
&q_eval/gradients/q_eval/Sum_grad/ShapeShape
q_eval/Mul*
T0*
out_type0*
_output_shapes
:
Ђ
%q_eval/gradients/q_eval/Sum_grad/SizeConst*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ь
$q_eval/gradients/q_eval/Sum_grad/addAddq_eval/Sum/reduction_indices%q_eval/gradients/q_eval/Sum_grad/Size*
_output_shapes
: *
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape
й
$q_eval/gradients/q_eval/Sum_grad/modFloorMod$q_eval/gradients/q_eval/Sum_grad/add%q_eval/gradients/q_eval/Sum_grad/Size*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
_output_shapes
: 
І
(q_eval/gradients/q_eval/Sum_grad/Shape_1Const*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
Љ
,q_eval/gradients/q_eval/Sum_grad/range/startConst*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Љ
,q_eval/gradients/q_eval/Sum_grad/range/deltaConst*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

&q_eval/gradients/q_eval/Sum_grad/rangeRange,q_eval/gradients/q_eval/Sum_grad/range/start%q_eval/gradients/q_eval/Sum_grad/Size,q_eval/gradients/q_eval/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape
Ј
+q_eval/gradients/q_eval/Sum_grad/Fill/valueConst*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ђ
%q_eval/gradients/q_eval/Sum_grad/FillFill(q_eval/gradients/q_eval/Sum_grad/Shape_1+q_eval/gradients/q_eval/Sum_grad/Fill/value*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*

index_type0*
_output_shapes
: 
Ю
.q_eval/gradients/q_eval/Sum_grad/DynamicStitchDynamicStitch&q_eval/gradients/q_eval/Sum_grad/range$q_eval/gradients/q_eval/Sum_grad/mod&q_eval/gradients/q_eval/Sum_grad/Shape%q_eval/gradients/q_eval/Sum_grad/Fill*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
N*#
_output_shapes
:џџџџџџџџџ
Ї
*q_eval/gradients/q_eval/Sum_grad/Maximum/yConst*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ј
(q_eval/gradients/q_eval/Sum_grad/MaximumMaximum.q_eval/gradients/q_eval/Sum_grad/DynamicStitch*q_eval/gradients/q_eval/Sum_grad/Maximum/y*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*#
_output_shapes
:џџџџџџџџџ
ч
)q_eval/gradients/q_eval/Sum_grad/floordivFloorDiv&q_eval/gradients/q_eval/Sum_grad/Shape(q_eval/gradients/q_eval/Sum_grad/Maximum*
T0*9
_class/
-+loc:@q_eval/gradients/q_eval/Sum_grad/Shape*
_output_shapes
:
б
(q_eval/gradients/q_eval/Sum_grad/ReshapeReshape;q_eval/gradients/q_eval/sub_grad/tuple/control_dependency_1.q_eval/gradients/q_eval/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
Ц
%q_eval/gradients/q_eval/Sum_grad/TileTile(q_eval/gradients/q_eval/Sum_grad/Reshape)q_eval/gradients/q_eval/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
p
&q_eval/gradients/q_eval/Mul_grad/ShapeShape
q_eval/add*
_output_shapes
:*
T0*
out_type0
{
(q_eval/gradients/q_eval/Mul_grad/Shape_1Shapeq_eval/action_taken*
T0*
out_type0*
_output_shapes
:
о
6q_eval/gradients/q_eval/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs&q_eval/gradients/q_eval/Mul_grad/Shape(q_eval/gradients/q_eval/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

$q_eval/gradients/q_eval/Mul_grad/MulMul%q_eval/gradients/q_eval/Sum_grad/Tileq_eval/action_taken*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
$q_eval/gradients/q_eval/Mul_grad/SumSum$q_eval/gradients/q_eval/Mul_grad/Mul6q_eval/gradients/q_eval/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
С
(q_eval/gradients/q_eval/Mul_grad/ReshapeReshape$q_eval/gradients/q_eval/Mul_grad/Sum&q_eval/gradients/q_eval/Mul_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

&q_eval/gradients/q_eval/Mul_grad/Mul_1Mul
q_eval/add%q_eval/gradients/q_eval/Sum_grad/Tile*'
_output_shapes
:џџџџџџџџџ*
T0
Я
&q_eval/gradients/q_eval/Mul_grad/Sum_1Sum&q_eval/gradients/q_eval/Mul_grad/Mul_18q_eval/gradients/q_eval/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*q_eval/gradients/q_eval/Mul_grad/Reshape_1Reshape&q_eval/gradients/q_eval/Mul_grad/Sum_1(q_eval/gradients/q_eval/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

1q_eval/gradients/q_eval/Mul_grad/tuple/group_depsNoOp)^q_eval/gradients/q_eval/Mul_grad/Reshape+^q_eval/gradients/q_eval/Mul_grad/Reshape_1

9q_eval/gradients/q_eval/Mul_grad/tuple/control_dependencyIdentity(q_eval/gradients/q_eval/Mul_grad/Reshape2^q_eval/gradients/q_eval/Mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_eval/gradients/q_eval/Mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;q_eval/gradients/q_eval/Mul_grad/tuple/control_dependency_1Identity*q_eval/gradients/q_eval/Mul_grad/Reshape_12^q_eval/gradients/q_eval/Mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/Mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
s
&q_eval/gradients/q_eval/add_grad/ShapeShapeq_eval/MatMul*
T0*
out_type0*
_output_shapes
:
r
(q_eval/gradients/q_eval/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
о
6q_eval/gradients/q_eval/add_grad/BroadcastGradientArgsBroadcastGradientArgs&q_eval/gradients/q_eval/add_grad/Shape(q_eval/gradients/q_eval/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
о
$q_eval/gradients/q_eval/add_grad/SumSum9q_eval/gradients/q_eval/Mul_grad/tuple/control_dependency6q_eval/gradients/q_eval/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(q_eval/gradients/q_eval/add_grad/ReshapeReshape$q_eval/gradients/q_eval/add_grad/Sum&q_eval/gradients/q_eval/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
т
&q_eval/gradients/q_eval/add_grad/Sum_1Sum9q_eval/gradients/q_eval/Mul_grad/tuple/control_dependency8q_eval/gradients/q_eval/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
К
*q_eval/gradients/q_eval/add_grad/Reshape_1Reshape&q_eval/gradients/q_eval/add_grad/Sum_1(q_eval/gradients/q_eval/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

1q_eval/gradients/q_eval/add_grad/tuple/group_depsNoOp)^q_eval/gradients/q_eval/add_grad/Reshape+^q_eval/gradients/q_eval/add_grad/Reshape_1

9q_eval/gradients/q_eval/add_grad/tuple/control_dependencyIdentity(q_eval/gradients/q_eval/add_grad/Reshape2^q_eval/gradients/q_eval/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_eval/gradients/q_eval/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;q_eval/gradients/q_eval/add_grad/tuple/control_dependency_1Identity*q_eval/gradients/q_eval/add_grad/Reshape_12^q_eval/gradients/q_eval/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/add_grad/Reshape_1*
_output_shapes
:
н
*q_eval/gradients/q_eval/MatMul_grad/MatMulMatMul9q_eval/gradients/q_eval/add_grad/tuple/control_dependencyq_eval/weights/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
з
,q_eval/gradients/q_eval/MatMul_grad/MatMul_1MatMulq_eval/strided_slice9q_eval/gradients/q_eval/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(

4q_eval/gradients/q_eval/MatMul_grad/tuple/group_depsNoOp+^q_eval/gradients/q_eval/MatMul_grad/MatMul-^q_eval/gradients/q_eval/MatMul_grad/MatMul_1

<q_eval/gradients/q_eval/MatMul_grad/tuple/control_dependencyIdentity*q_eval/gradients/q_eval/MatMul_grad/MatMul5^q_eval/gradients/q_eval/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

>q_eval/gradients/q_eval/MatMul_grad/tuple/control_dependency_1Identity,q_eval/gradients/q_eval/MatMul_grad/MatMul_15^q_eval/gradients/q_eval/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*?
_class5
31loc:@q_eval/gradients/q_eval/MatMul_grad/MatMul_1

0q_eval/gradients/q_eval/strided_slice_grad/ShapeShapeq_eval/rnn/transpose_1*
T0*
out_type0*
_output_shapes
:
Ш
;q_eval/gradients/q_eval/strided_slice_grad/StridedSliceGradStridedSliceGrad0q_eval/gradients/q_eval/strided_slice_grad/Shapeq_eval/strided_slice/stackq_eval/strided_slice/stack_1q_eval/strided_slice/stack_2<q_eval/gradients/q_eval/MatMul_grad/tuple/control_dependency*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ

>q_eval/gradients/q_eval/rnn/transpose_1_grad/InvertPermutationInvertPermutationq_eval/rnn/concat_2*
T0*
_output_shapes
:

6q_eval/gradients/q_eval/rnn/transpose_1_grad/transpose	Transpose;q_eval/gradients/q_eval/strided_slice_grad/StridedSliceGrad>q_eval/gradients/q_eval/rnn/transpose_1_grad/InvertPermutation*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
Tperm0

gq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3q_eval/rnn/TensorArrayq_eval/rnn/while/Exit_2*)
_class
loc:@q_eval/rnn/TensorArray*
sourceq_eval/gradients*
_output_shapes

:: 
О
cq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityq_eval/rnn/while/Exit_2h^q_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*)
_class
loc:@q_eval/rnn/TensorArray*
_output_shapes
: 
Я
mq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3gq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3!q_eval/rnn/TensorArrayStack/range6q_eval/gradients/q_eval/rnn/transpose_1_grad/transposecq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
t
q_eval/gradients/zeros_like	ZerosLikeq_eval/rnn/while/Exit_3*
T0*(
_output_shapes
:џџџџџџџџџ
v
q_eval/gradients/zeros_like_1	ZerosLikeq_eval/rnn/while/Exit_4*
T0*(
_output_shapes
:џџџџџџџџџ
М
4q_eval/gradients/q_eval/rnn/while/Exit_2_grad/b_exitEntermq_eval/gradients/q_eval/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ќ
4q_eval/gradients/q_eval/rnn/while/Exit_3_grad/b_exitEnterq_eval/gradients/zeros_like*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ў
4q_eval/gradients/q_eval/rnn/while/Exit_4_grad/b_exitEnterq_eval/gradients/zeros_like_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ф
8q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switchMerge4q_eval/gradients/q_eval/rnn/while/Exit_2_grad/b_exit?q_eval/gradients/q_eval/rnn/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
і
8q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switchMerge4q_eval/gradients/q_eval/rnn/while/Exit_3_grad/b_exit?q_eval/gradients/q_eval/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
і
8q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switchMerge4q_eval/gradients/q_eval/rnn/while/Exit_4_grad/b_exit?q_eval/gradients/q_eval/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:џџџџџџџџџ: 

5q_eval/gradients/q_eval/rnn/while/Merge_2_grad/SwitchSwitch8q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switchq_eval/gradients/b_count_2*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 

?q_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/group_depsNoOp6^q_eval/gradients/q_eval/rnn/while/Merge_2_grad/Switch
К
Gq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity5q_eval/gradients/q_eval/rnn/while/Merge_2_grad/Switch@^q_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
О
Iq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity7q_eval/gradients/q_eval/rnn/while/Merge_2_grad/Switch:1@^q_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/group_deps*
_output_shapes
: *
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switch
Љ
5q_eval/gradients/q_eval/rnn/while/Merge_3_grad/SwitchSwitch8q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switchq_eval/gradients/b_count_2*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switch*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

?q_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/group_depsNoOp6^q_eval/gradients/q_eval/rnn/while/Merge_3_grad/Switch
Ь
Gq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity5q_eval/gradients/q_eval/rnn/while/Merge_3_grad/Switch@^q_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
а
Iq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity7q_eval/gradients/q_eval/rnn/while/Merge_3_grad/Switch:1@^q_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
Љ
5q_eval/gradients/q_eval/rnn/while/Merge_4_grad/SwitchSwitch8q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switchq_eval/gradients/b_count_2*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switch*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

?q_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/group_depsNoOp6^q_eval/gradients/q_eval/rnn/while/Merge_4_grad/Switch
Ь
Gq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity5q_eval/gradients/q_eval/rnn/while/Merge_4_grad/Switch@^q_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
а
Iq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity7q_eval/gradients/q_eval/rnn/while/Merge_4_grad/Switch:1@^q_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
Ѕ
3q_eval/gradients/q_eval/rnn/while/Enter_2_grad/ExitExitGq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency*
_output_shapes
: *
T0
З
3q_eval/gradients/q_eval/rnn/while/Enter_3_grad/ExitExitGq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
З
3q_eval/gradients/q_eval/rnn/while/Enter_4_grad/ExitExitGq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Б
lq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterIq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1*
_output_shapes

:: *3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
sourceq_eval/gradients
м
rq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterq_eval/rnn/TensorArray*
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

hq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityIq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1m^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*3
_class)
'%loc:@q_eval/rnn/while/lstm_cell/mul_2
щ
\q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3lq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3gq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2hq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:џџџџџџџџџ
н
bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*.
_class$
" loc:@q_eval/rnn/while/Identity_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
_output_shapes
:*
	elem_type0*.
_class$
" loc:@q_eval/rnn/while/Identity_1
в
bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterbq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
У
hq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2bq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterq_eval/rnn/while/Identity_1^q_eval/gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
Є
gq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2mq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^q_eval/gradients/Sub*
_output_shapes
: *
	elem_type0
ю
mq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterbq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
х
cq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerB^q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2F^q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2F^q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2h^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2L^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2X^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2Z^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1V^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2J^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2X^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2Z^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2H^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2X^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2Z^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2H^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2V^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2X^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2

[q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpJ^q_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1]^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Я
cq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentity\q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3\^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*o
_classe
caloc:@q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*(
_output_shapes
:џџџџџџџџџ

eq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityIq_eval/gradients/q_eval/rnn/while/Merge_2_grad/tuple/control_dependency_1\^q_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
С
:q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like	ZerosLikeEq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
Л
@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/ConstConst*.
_class$
" loc:@q_eval/rnn/while/Identity_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ќ
@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/f_accStackV2@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/Const*.
_class$
" loc:@q_eval/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0

@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/EnterEnter@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/Enterq_eval/rnn/while/Identity_3^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
Н
6q_eval/gradients/q_eval/rnn/while/Select_1_grad/SelectSelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2Iq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependency_1:q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
Й
<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/ConstConst*0
_class&
$"loc:@q_eval/rnn/while/GreaterEqual*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
і
<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/f_accStackV2<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/Const*
	elem_type0
*0
_class&
$"loc:@q_eval/rnn/while/GreaterEqual*

stack_name *
_output_shapes
:

<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/EnterEnter<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
ћ
Bq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPushV2StackPushV2<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/Enterq_eval/rnn/while/GreaterEqual^q_eval/gradients/Add*
T0
*
_output_shapes
:*
swap_memory( 
к
Aq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2
StackPopV2Gq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0

Ђ
Gq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2/EnterEnter<q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
П
8q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1SelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2:q_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_likeIq_eval/gradients/q_eval/rnn/while/Merge_3_grad/tuple/control_dependency_1*(
_output_shapes
:џџџџџџџџџ*
T0
М
@q_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/group_depsNoOp7^q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select9^q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1
Э
Hq_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/control_dependencyIdentity6q_eval/gradients/q_eval/rnn/while/Select_1_grad/SelectA^q_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select
г
Jq_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/control_dependency_1Identity8q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1A^q_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1
С
:q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like	ZerosLikeEq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
Л
@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/ConstConst*.
_class$
" loc:@q_eval/rnn/while/Identity_4*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ќ
@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/f_accStackV2@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/Const*.
_class$
" loc:@q_eval/rnn/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0

@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/EnterEnter@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/Enterq_eval/rnn/while/Identity_4^q_eval/gradients/Add*(
_output_shapes
:џџџџџџџџџ*
swap_memory( *
T0
ђ
Eq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
Н
6q_eval/gradients/q_eval/rnn/while/Select_2_grad/SelectSelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2Iq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependency_1:q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
П
8q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1SelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2:q_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_likeIq_eval/gradients/q_eval/rnn/while/Merge_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
М
@q_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/group_depsNoOp7^q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select9^q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1
Э
Hq_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/control_dependencyIdentity6q_eval/gradients/q_eval/rnn/while/Select_2_grad/SelectA^q_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select*(
_output_shapes
:џџџџџџџџџ
г
Jq_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/control_dependency_1Identity8q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1A^q_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1*(
_output_shapes
:џџџџџџџџџ
Я
8q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_like	ZerosLike>q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_like/Enter^q_eval/gradients/Sub*
T0*(
_output_shapes
:џџџџџџџџџ
ћ
>q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_like/EnterEnterq_eval/rnn/zeros*
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
г
4q_eval/gradients/q_eval/rnn/while/Select_grad/SelectSelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV2cq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency8q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
е
6q_eval/gradients/q_eval/rnn/while/Select_grad/Select_1SelectAq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPopV28q_eval/gradients/q_eval/rnn/while/Select_grad/zeros_likecq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
>q_eval/gradients/q_eval/rnn/while/Select_grad/tuple/group_depsNoOp5^q_eval/gradients/q_eval/rnn/while/Select_grad/Select7^q_eval/gradients/q_eval/rnn/while/Select_grad/Select_1
Х
Fq_eval/gradients/q_eval/rnn/while/Select_grad/tuple/control_dependencyIdentity4q_eval/gradients/q_eval/rnn/while/Select_grad/Select?^q_eval/gradients/q_eval/rnn/while/Select_grad/tuple/group_deps*
T0*G
_class=
;9loc:@q_eval/gradients/q_eval/rnn/while/Select_grad/Select*(
_output_shapes
:џџџџџџџџџ
Ы
Hq_eval/gradients/q_eval/rnn/while/Select_grad/tuple/control_dependency_1Identity6q_eval/gradients/q_eval/rnn/while/Select_grad/Select_1?^q_eval/gradients/q_eval/rnn/while/Select_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_grad/Select_1*(
_output_shapes
:џџџџџџџџџ

9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/ShapeShapeq_eval/rnn/zeros*
T0*
out_type0*
_output_shapes
:

?q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/zerosFill9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/Shape?q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0

9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_accEnter9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/zeros*
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant( 

;q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_acc_1Merge9q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_accAq_eval/gradients/q_eval/rnn/while/Select/Enter_grad/NextIteration*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
ф
:q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/SwitchSwitch;q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_acc_1q_eval/gradients/b_count_2*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0
ї
7q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/AddAdd<q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/Switch:1Fq_eval/gradients/q_eval/rnn/while/Select_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
О
Aq_eval/gradients/q_eval/rnn/while/Select/Enter_grad/NextIterationNextIteration7q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/Add*(
_output_shapes
:џџџџџџџџџ*
T0
В
;q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_acc_2Exit:q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/Switch*
T0*(
_output_shapes
:џџџџџџџџџ
М
q_eval/gradients/AddNAddNJq_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/control_dependency_1Hq_eval/gradients/q_eval/rnn/while/Select_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select_1*
N*(
_output_shapes
:џџџџџџџџџ
 
<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/ShapeShape$q_eval/rnn/while/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape_1Shape!q_eval/rnn/while/lstm_cell/Tanh_1*
_output_shapes
:*
T0*
out_type0
ж
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsWq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ю
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape
В
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ш
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
Ю
]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ђ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape_1
Ж
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1EnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ю
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape_1^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_eval/gradients/Sub*
	elem_type0*
_output_shapes
:
в
_q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
в
:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/MulMulq_eval/gradients/AddNEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
С
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/ConstConst*4
_class*
(&loc:@q_eval/rnn/while/lstm_cell/Tanh_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/f_accStackV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/Const*
	elem_type0*4
_class*
(&loc:@q_eval/rnn/while/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/Enter!q_eval/rnn/while/lstm_cell/Tanh_1^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/SumSum:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/MulLq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/ReshapeReshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/SumWq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
ж
<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1MulGq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2q_eval/gradients/AddN*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*7
_class-
+)loc:@q_eval/rnn/while/lstm_cell/Sigmoid_2*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/Const*7
_class-
+)loc:@q_eval/rnn/while/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterBq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Hq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter$q_eval/rnn/while/lstm_cell/Sigmoid_2^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
і
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0*(
_output_shapes
:џџџџџџџџџ
Ў
Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterBq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Sum_1Sum<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1Nq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape_1Reshape<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Sum_1Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
г
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/ReshapeA^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape_1
ы
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentity>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/ReshapeH^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1Identity@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape_1H^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Reshape_1
и
?q_eval/gradients/q_eval/rnn/while/Switch_2_grad_1/NextIterationNextIterationeq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
}
,q_eval/gradients/q_eval/rnn/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
к
*q_eval/gradients/q_eval/rnn/zeros_grad/SumSum;q_eval/gradients/q_eval/rnn/while/Select/Enter_grad/b_acc_2,q_eval/gradients/q_eval/rnn/zeros_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Ђ
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradGq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

@q_eval/gradients/q_eval/rnn/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
q_eval/gradients/AddN_1AddNJq_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/control_dependency_1@q_eval/gradients/q_eval/rnn/while/lstm_cell/Tanh_1_grad/TanhGrad*
N*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select_1

<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/ShapeShapeq_eval/rnn/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape_1Shape q_eval/rnn/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
ж
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsWq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2Yq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ю
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
В
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ш
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
]q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
ђ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape_1*
valueB :
џџџџџџџџџ
Ч
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:
Ж
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1EnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ю
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape_1^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
в
_q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
ш
:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/SumSumq_eval/gradients/AddN_1Lq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/ReshapeReshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/SumWq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
ь
<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Sum_1Sumq_eval/gradients/AddN_1Nq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape_1Reshape<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Sum_1Yq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/ReshapeA^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape_1
ы
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentity>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/ReshapeH^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1Identity@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape_1H^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Reshape_1

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/ShapeShape"q_eval/rnn/while/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape_1Shapeq_eval/rnn/while/Identity_3*
_output_shapes
:*
T0*
out_type0
а
Jq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsUq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ъ
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*M
_classC
A?loc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*
	elem_type0*M
_classC
A?loc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:
Ў
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterPq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant(
Т
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Uq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2[q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0*
_output_shapes
:
Ъ
[q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterPq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(
ю
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:
В
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1EnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant(
Ш
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape_1^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
Ю
]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(

8q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/MulMulOq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyEq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0

8q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/SumSum8q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/MulJq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/ReshapeReshape8q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/SumUq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1MulEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2Oq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
Т
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/ConstConst*5
_class+
)'loc:@q_eval/rnn/while/lstm_cell/Sigmoid*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/f_accStackV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/Const*
	elem_type0*5
_class+
)'loc:@q_eval/rnn/while/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/Enter"q_eval/rnn/while/lstm_cell/Sigmoid^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0*(
_output_shapes
:џџџџџџџџџ
Њ
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Sum_1Sum:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1Lq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape_1Reshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Sum_1Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Э
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/group_depsNoOp=^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape?^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape_1
у
Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/control_dependencyIdentity<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/ReshapeF^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
щ
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1Identity>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
 
<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/ShapeShape$q_eval/rnn/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape_1Shapeq_eval/rnn/while/lstm_cell/Tanh*
_output_shapes
:*
T0*
out_type0
ж
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsWq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ю
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape*

stack_name *
_output_shapes
:
В
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ш
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
Ю
]q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
ђ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:
Ж
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1EnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
Ю
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape_1^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
в
_q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/MulMulQq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0
П
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/ConstConst*2
_class(
&$loc:@q_eval/rnn/while/lstm_cell/Tanh*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/f_accStackV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/Const*2
_class(
&$loc:@q_eval/rnn/while/lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0

@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/Enterq_eval/rnn/while/lstm_cell/Tanh^q_eval/gradients/Add*(
_output_shapes
:џџџџџџџџџ*
swap_memory( *
T0
ђ
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnter@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*
is_constant(

:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/SumSum:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/MulLq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/ReshapeReshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/SumWq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1MulGq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *7
_class-
+)loc:@q_eval/rnn/while/lstm_cell/Sigmoid_1*
valueB :
џџџџџџџџџ

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*7
_class-
+)loc:@q_eval/rnn/while/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterBq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context

Hq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter$q_eval/rnn/while/lstm_cell/Sigmoid_1^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
і
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ў
Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterBq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Sum_1Sum<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1Nq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѕ
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape_1Reshape<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Sum_1Yq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Gq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/ReshapeA^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape_1
ы
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentity>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/ReshapeH^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape
ё
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1Identity@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape_1H^q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2Mq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
С
q_eval/gradients/AddN_2AddNHq_eval/gradients/q_eval/rnn/while/Select_1_grad/tuple/control_dependencyOq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select
Ђ
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradGq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Oq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

>q_eval/gradients/q_eval/rnn/while/lstm_cell/Tanh_grad/TanhGradTanhGradEq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ

:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/ShapeShape"q_eval/rnn/while/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:

<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape_1Const^q_eval/gradients/Sub*
dtype0*
_output_shapes
: *
valueB 
Е
Jq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsUq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*M
_classC
A?loc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*M
_classC
A?loc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape
Ў
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterPq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant(
Т
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape^q_eval/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Uq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2[q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^q_eval/gradients/Sub*
_output_shapes
:*
	elem_type0
Ъ
[q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterPq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

8q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/SumSumDq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradJq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/ReshapeReshape8q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/SumUq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Sum_1SumDq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradLq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ђ
>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape_1Reshape:q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Sum_1<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Э
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/group_depsNoOp=^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape?^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape_1
у
Mq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/control_dependencyIdentity<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/ReshapeF^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
з
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/control_dependency_1Identity>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape_1F^q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 

?q_eval/gradients/q_eval/rnn/while/Switch_3_grad_1/NextIterationNextIterationq_eval/gradients/AddN_2*
T0*(
_output_shapes
:џџџџџџџџџ
ѕ
=q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concatConcatV2Fq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGrad>q_eval/gradients/q_eval/rnn/while/lstm_cell/Tanh_grad/TanhGradMq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/tuple/control_dependencyFq_eval/gradients/q_eval/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradCq_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat/Const*
N*(
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0

Cq_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat/ConstConst^q_eval/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
Я
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad=q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:*
T0
и
Iq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpE^q_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad>^q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat
э
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity=q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concatJ^q_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/split_grad/concat*(
_output_shapes
:џџџџџџџџџ
№
Sq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityDq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradJ^q_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
К
>q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMulMatMulQq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyDq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul/Enter*
T0*(
_output_shapes
:џџџџџџџџџ *
transpose_a( *
transpose_b(

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul/EnterEnter q_eval/rnn/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
 *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
Л
@q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1MatMulKq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
 *
transpose_a(*
transpose_b( *
T0
Ч
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@q_eval/rnn/while/lstm_cell/concat*
valueB :
џџџџџџџџџ

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Const*4
_class*
(&loc:@q_eval/rnn/while/lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterFq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context*
T0*
is_constant(
Ѓ
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Fq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter!q_eval/rnn/while/lstm_cell/concat^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ *
swap_memory( 
ў
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Qq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^q_eval/gradients/Sub*
	elem_type0*(
_output_shapes
:џџџџџџџџџ 
Ж
Qq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterFq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
д
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMulA^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1
э
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentity>q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMulI^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ 
ы
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1Identity@q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1I^q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
 *
T0*S
_classI
GEloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes	
:*
valueB*    
Њ
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterDq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

Fq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeFq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Lq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
р
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchFq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2q_eval/gradients/b_count_2*
T0*"
_output_shapes
::

Bq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/AddAddGq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch:1Sq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ч
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationBq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
Л
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitEq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:

=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ConstConst^q_eval/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

<q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/RankConst^q_eval/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
х
;q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/modFloorMod=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Const<q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 

=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeShape"q_eval/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:

>q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeNShapeNIq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2Eq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPopV2*
T0*
out_type0*
N* 
_output_shapes
::
Ц
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/ConstConst*5
_class+
)'loc:@q_eval/rnn/while/TensorArrayReadV3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/Const*

stack_name *
_output_shapes
:*
	elem_type0*5
_class+
)'loc:@q_eval/rnn/while/TensorArrayReadV3

Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/EnterEnterDq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_eval/rnn/while/while_context
 
Jq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/Enter"q_eval/rnn/while/TensorArrayReadV3^q_eval/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ *
swap_memory( 
њ
Iq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Oq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^q_eval/gradients/Sub*(
_output_shapes
:џџџџџџџџџ *
	elem_type0
В
Oq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterDq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context
О
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ConcatOffsetConcatOffset;q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/mod>q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN@q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
ц
=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/SliceSlicePq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyDq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ConcatOffset>q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ь
?q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice_1SlicePq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyFq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ConcatOffset:1@q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Index0*
T0
в
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/group_depsNoOp>^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice@^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice_1
ы
Pq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyIdentity=q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/SliceI^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice*(
_output_shapes
:џџџџџџџџџ 
ё
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1Identity?q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice_1I^q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*R
_classH
FDloc:@q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Slice_1*(
_output_shapes
:џџџџџџџџџ

Cq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
 *    *
dtype0* 
_output_shapes
:
 
­
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterCq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
 *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

Eq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeEq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1Kq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
 : *
T0
ш
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchEq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2q_eval/gradients/b_count_2*
T0*,
_output_shapes
:
 :
 

Aq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/AddAddFq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/Switch:1Rq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
 
Ъ
Kq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationAq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/Add* 
_output_shapes
:
 *
T0
О
Eq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitDq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
 
Х
Zq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3`q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterbq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^q_eval/gradients/Sub*;
_class1
/-loc:@q_eval/rnn/while/TensorArrayReadV3/Enter*
sourceq_eval/gradients*
_output_shapes

:: 
д
`q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterq_eval/rnn/TensorArray_1*
_output_shapes
:*?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*;
_class1
/-loc:@q_eval/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(
џ
bq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterEq_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context*
T0*;
_class1
/-loc:@q_eval/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 

Vq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitybq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1[^q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*;
_class1
/-loc:@q_eval/rnn/while/TensorArrayReadV3/Enter

\q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Zq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3gq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Pq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyVq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
Ф
q_eval/gradients/AddN_3AddNHq_eval/gradients/q_eval/rnn/while/Select_2_grad/tuple/control_dependencyRq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/rnn/while/Select_2_grad/Select*
N*(
_output_shapes
:џџџџџџџџџ

Fq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*
_output_shapes
: *
valueB
 *    
Љ
Hq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterFq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *?

frame_name1/q_eval/gradients/q_eval/rnn/while/while_context

Hq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeHq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Nq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*
_output_shapes
: : *
T0
к
Gq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchHq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2q_eval/gradients/b_count_2*
T0*
_output_shapes
: : 

Dq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddIq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1\q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ц
Nq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationDq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
К
Hq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitGq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 

?q_eval/gradients/q_eval/rnn/while/Switch_4_grad_1/NextIterationNextIterationq_eval/gradients/AddN_3*(
_output_shapes
:џџџџџџџџџ*
T0
п
}q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3q_eval/rnn/TensorArray_1Hq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*+
_class!
loc:@q_eval/rnn/TensorArray_1*
sourceq_eval/gradients*
_output_shapes

:: 

yq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityHq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3~^q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@q_eval/rnn/TensorArray_1*
_output_shapes
: 

oq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3}q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3#q_eval/rnn/TensorArrayUnstack/rangeyq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ 
Б
lq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpp^q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3I^q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
Ѕ
tq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityoq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3m^q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
T0*
_classx
vtloc:@q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
Й
vq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityHq_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3m^q_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@q_eval/gradients/q_eval/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 

<q_eval/gradients/q_eval/rnn/transpose_grad/InvertPermutationInvertPermutationq_eval/rnn/concat*
T0*
_output_shapes
:
Т
4q_eval/gradients/q_eval/rnn/transpose_grad/transpose	Transposetq_eval/gradients/q_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency<q_eval/gradients/q_eval/rnn/transpose_grad/InvertPermutation*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
Tperm0*
T0
z
,q_eval/gradients/q_eval/Reshape_1_grad/ShapeShapeq_eval/Reshape*
T0*
out_type0*
_output_shapes
:
о
.q_eval/gradients/q_eval/Reshape_1_grad/ReshapeReshape4q_eval/gradients/q_eval/rnn/transpose_grad/transpose,q_eval/gradients/q_eval/Reshape_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ 
w
*q_eval/gradients/q_eval/Reshape_grad/ShapeShapeq_eval/Relu_1*
T0*
out_type0*
_output_shapes
:
л
,q_eval/gradients/q_eval/Reshape_grad/ReshapeReshape.q_eval/gradients/q_eval/Reshape_1_grad/Reshape*q_eval/gradients/q_eval/Reshape_grad/Shape*/
_output_shapes
:џџџџџџџџџ		 *
T0*
Tshape0
Џ
,q_eval/gradients/q_eval/Relu_1_grad/ReluGradReluGrad,q_eval/gradients/q_eval/Reshape_grad/Reshapeq_eval/Relu_1*/
_output_shapes
:џџџџџџџџџ		 *
T0
Џ
6q_eval/gradients/q_eval/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad,q_eval/gradients/q_eval/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
Ћ
;q_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/group_depsNoOp-^q_eval/gradients/q_eval/Relu_1_grad/ReluGrad7^q_eval/gradients/q_eval/conv2/BiasAdd_grad/BiasAddGrad
Ж
Cq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependencyIdentity,q_eval/gradients/q_eval/Relu_1_grad/ReluGrad<^q_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@q_eval/gradients/q_eval/Relu_1_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ		 
З
Eq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependency_1Identity6q_eval/gradients/q_eval/conv2/BiasAdd_grad/BiasAddGrad<^q_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/conv2/BiasAdd_grad/BiasAddGrad
Ѕ
0q_eval/gradients/q_eval/conv2/Conv2D_grad/ShapeNShapeNq_eval/Reluq_eval/conv2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::

/q_eval/gradients/q_eval/conv2/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Љ
=q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput0q_eval/gradients/q_eval/conv2/Conv2D_grad/ShapeNq_eval/conv2/kernel/readCq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
љ
>q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterq_eval/Relu/q_eval/gradients/q_eval/conv2/Conv2D_grad/ConstCq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
: 
У
:q_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropFilter>^q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropInput
ж
Bq_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/control_dependencyIdentity=q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropInput;^q_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ
б
Dq_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/control_dependency_1Identity>q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropFilter;^q_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
С
*q_eval/gradients/q_eval/Relu_grad/ReluGradReluGradBq_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/control_dependencyq_eval/Relu*
T0*/
_output_shapes
:џџџџџџџџџ
­
6q_eval/gradients/q_eval/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad*q_eval/gradients/q_eval/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
Љ
;q_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/group_depsNoOp+^q_eval/gradients/q_eval/Relu_grad/ReluGrad7^q_eval/gradients/q_eval/conv1/BiasAdd_grad/BiasAddGrad
В
Cq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependencyIdentity*q_eval/gradients/q_eval/Relu_grad/ReluGrad<^q_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_eval/gradients/q_eval/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ
З
Eq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependency_1Identity6q_eval/gradients/q_eval/conv1/BiasAdd_grad/BiasAddGrad<^q_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_eval/gradients/q_eval/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ї
0q_eval/gradients/q_eval/conv1/Conv2D_grad/ShapeNShapeNq_eval/statesq_eval/conv1/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0

/q_eval/gradients/q_eval/conv1/Conv2D_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
Љ
=q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput0q_eval/gradients/q_eval/conv1/Conv2D_grad/ShapeNq_eval/conv1/kernel/readCq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ћ
>q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterq_eval/states/q_eval/gradients/q_eval/conv1/Conv2D_grad/ConstCq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
У
:q_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/group_depsNoOp?^q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropFilter>^q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropInput
ж
Bq_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/control_dependencyIdentity=q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropInput;^q_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџTT*
T0*P
_classF
DBloc:@q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropInput
б
Dq_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/control_dependency_1Identity>q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropFilter;^q_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_eval/gradients/q_eval/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

 q_eval/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: * 
_class
loc:@q_eval/biases*
valueB
 *fff?

q_eval/beta1_power
VariableV2*
shared_name * 
_class
loc:@q_eval/biases*
	container *
shape: *
dtype0*
_output_shapes
: 
Х
q_eval/beta1_power/AssignAssignq_eval/beta1_power q_eval/beta1_power/initial_value*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: 
z
q_eval/beta1_power/readIdentityq_eval/beta1_power*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
: 

 q_eval/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: * 
_class
loc:@q_eval/biases*
valueB
 *wО?

q_eval/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@q_eval/biases*
	container *
shape: 
Х
q_eval/beta2_power/AssignAssignq_eval/beta2_power q_eval/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@q_eval/biases
z
q_eval/beta2_power/readIdentityq_eval/beta2_power*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
: 
Т
Aq_eval/q_eval/conv1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_eval/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Є
7q_eval/q_eval/conv1/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@q_eval/conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
 
1q_eval/q_eval/conv1/kernel/Adam/Initializer/zerosFillAq_eval/q_eval/conv1/kernel/Adam/Initializer/zeros/shape_as_tensor7q_eval/q_eval/conv1/kernel/Adam/Initializer/zeros/Const*&
_output_shapes
:*
T0*&
_class
loc:@q_eval/conv1/kernel*

index_type0
Ы
q_eval/q_eval/conv1/kernel/Adam
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *&
_class
loc:@q_eval/conv1/kernel*
	container *
shape:

&q_eval/q_eval/conv1/kernel/Adam/AssignAssignq_eval/q_eval/conv1/kernel/Adam1q_eval/q_eval/conv1/kernel/Adam/Initializer/zeros*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
Њ
$q_eval/q_eval/conv1/kernel/Adam/readIdentityq_eval/q_eval/conv1/kernel/Adam*
T0*&
_class
loc:@q_eval/conv1/kernel*&
_output_shapes
:
Ф
Cq_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_eval/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
І
9q_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@q_eval/conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
І
3q_eval/q_eval/conv1/kernel/Adam_1/Initializer/zerosFillCq_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensor9q_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@q_eval/conv1/kernel*

index_type0*&
_output_shapes
:
Э
!q_eval/q_eval/conv1/kernel/Adam_1
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *&
_class
loc:@q_eval/conv1/kernel*
	container *
shape:

(q_eval/q_eval/conv1/kernel/Adam_1/AssignAssign!q_eval/q_eval/conv1/kernel/Adam_13q_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel
Ў
&q_eval/q_eval/conv1/kernel/Adam_1/readIdentity!q_eval/q_eval/conv1/kernel/Adam_1*&
_output_shapes
:*
T0*&
_class
loc:@q_eval/conv1/kernel
Ђ
/q_eval/q_eval/conv1/bias/Adam/Initializer/zerosConst*$
_class
loc:@q_eval/conv1/bias*
valueB*    *
dtype0*
_output_shapes
:
Џ
q_eval/q_eval/conv1/bias/Adam
VariableV2*
shared_name *$
_class
loc:@q_eval/conv1/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ђ
$q_eval/q_eval/conv1/bias/Adam/AssignAssignq_eval/q_eval/conv1/bias/Adam/q_eval/q_eval/conv1/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:

"q_eval/q_eval/conv1/bias/Adam/readIdentityq_eval/q_eval/conv1/bias/Adam*
T0*$
_class
loc:@q_eval/conv1/bias*
_output_shapes
:
Є
1q_eval/q_eval/conv1/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@q_eval/conv1/bias*
valueB*    *
dtype0*
_output_shapes
:
Б
q_eval/q_eval/conv1/bias/Adam_1
VariableV2*$
_class
loc:@q_eval/conv1/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ј
&q_eval/q_eval/conv1/bias/Adam_1/AssignAssignq_eval/q_eval/conv1/bias/Adam_11q_eval/q_eval/conv1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias

$q_eval/q_eval/conv1/bias/Adam_1/readIdentityq_eval/q_eval/conv1/bias/Adam_1*
T0*$
_class
loc:@q_eval/conv1/bias*
_output_shapes
:
Т
Aq_eval/q_eval/conv2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*&
_class
loc:@q_eval/conv2/kernel*%
valueB"             
Є
7q_eval/q_eval/conv2/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@q_eval/conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
 
1q_eval/q_eval/conv2/kernel/Adam/Initializer/zerosFillAq_eval/q_eval/conv2/kernel/Adam/Initializer/zeros/shape_as_tensor7q_eval/q_eval/conv2/kernel/Adam/Initializer/zeros/Const*&
_output_shapes
: *
T0*&
_class
loc:@q_eval/conv2/kernel*

index_type0
Ы
q_eval/q_eval/conv2/kernel/Adam
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_eval/conv2/kernel*
	container *
shape: 

&q_eval/q_eval/conv2/kernel/Adam/AssignAssignq_eval/q_eval/conv2/kernel/Adam1q_eval/q_eval/conv2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: 
Њ
$q_eval/q_eval/conv2/kernel/Adam/readIdentityq_eval/q_eval/conv2/kernel/Adam*&
_output_shapes
: *
T0*&
_class
loc:@q_eval/conv2/kernel
Ф
Cq_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*&
_class
loc:@q_eval/conv2/kernel*%
valueB"             
І
9q_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_eval/conv2/kernel*
valueB
 *    
І
3q_eval/q_eval/conv2/kernel/Adam_1/Initializer/zerosFillCq_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensor9q_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros/Const*&
_output_shapes
: *
T0*&
_class
loc:@q_eval/conv2/kernel*

index_type0
Э
!q_eval/q_eval/conv2/kernel/Adam_1
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_eval/conv2/kernel*
	container *
shape: 

(q_eval/q_eval/conv2/kernel/Adam_1/AssignAssign!q_eval/q_eval/conv2/kernel/Adam_13q_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel
Ў
&q_eval/q_eval/conv2/kernel/Adam_1/readIdentity!q_eval/q_eval/conv2/kernel/Adam_1*&
_output_shapes
: *
T0*&
_class
loc:@q_eval/conv2/kernel
Ђ
/q_eval/q_eval/conv2/bias/Adam/Initializer/zerosConst*$
_class
loc:@q_eval/conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Џ
q_eval/q_eval/conv2/bias/Adam
VariableV2*
shared_name *$
_class
loc:@q_eval/conv2/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ђ
$q_eval/q_eval/conv2/bias/Adam/AssignAssignq_eval/q_eval/conv2/bias/Adam/q_eval/q_eval/conv2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias

"q_eval/q_eval/conv2/bias/Adam/readIdentityq_eval/q_eval/conv2/bias/Adam*
T0*$
_class
loc:@q_eval/conv2/bias*
_output_shapes
: 
Є
1q_eval/q_eval/conv2/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@q_eval/conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Б
q_eval/q_eval/conv2/bias/Adam_1
VariableV2*
shared_name *$
_class
loc:@q_eval/conv2/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ј
&q_eval/q_eval/conv2/bias/Adam_1/AssignAssignq_eval/q_eval/conv2/bias/Adam_11q_eval/q_eval/conv2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: 

$q_eval/q_eval/conv2/bias/Adam_1/readIdentityq_eval/q_eval/conv2/bias/Adam_1*
T0*$
_class
loc:@q_eval/conv2/bias*
_output_shapes
: 
Ъ
Iq_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Д
?q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
К
9q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zerosFillIq_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensor?q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*

index_type0* 
_output_shapes
:
 
Я
'q_eval/q_eval/rnn/lstm_cell/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
 *
shared_name *.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
	container *
shape:
 
 
.q_eval/q_eval/rnn/lstm_cell/kernel/Adam/AssignAssign'q_eval/q_eval/rnn/lstm_cell/kernel/Adam9q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
М
,q_eval/q_eval/rnn/lstm_cell/kernel/Adam/readIdentity'q_eval/q_eval/rnn/lstm_cell/kernel/Adam*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel* 
_output_shapes
:
 
Ь
Kq_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ж
Aq_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
;q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zerosFillKq_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorAq_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
 *
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*

index_type0
б
)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1
VariableV2*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
	container *
shape:
 *
dtype0* 
_output_shapes
:
 *
shared_name 
І
0q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/AssignAssign)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1;q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
Р
.q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/readIdentity)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel* 
_output_shapes
:
 
Р
Gq_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
А
=q_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros/ConstConst*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
­
7q_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zerosFillGq_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensor=q_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
С
%q_eval/q_eval/rnn/lstm_cell/bias/Adam
VariableV2*
shared_name *,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
	container *
shape:*
dtype0*
_output_shapes	
:

,q_eval/q_eval/rnn/lstm_cell/bias/Adam/AssignAssign%q_eval/q_eval/rnn/lstm_cell/bias/Adam7q_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias
Б
*q_eval/q_eval/rnn/lstm_cell/bias/Adam/readIdentity%q_eval/q_eval/rnn/lstm_cell/bias/Adam*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
_output_shapes	
:
Т
Iq_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
В
?q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
valueB
 *    
Г
9q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zerosFillIq_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensor?q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
У
'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1
VariableV2*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

.q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/AssignAssign'q_eval/q_eval/rnn/lstm_cell/bias/Adam_19q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Е
,q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/readIdentity'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
_output_shapes	
:
А
<q_eval/q_eval/weights/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*!
_class
loc:@q_eval/weights*
valueB"      

2q_eval/q_eval/weights/Adam/Initializer/zeros/ConstConst*!
_class
loc:@q_eval/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

,q_eval/q_eval/weights/Adam/Initializer/zerosFill<q_eval/q_eval/weights/Adam/Initializer/zeros/shape_as_tensor2q_eval/q_eval/weights/Adam/Initializer/zeros/Const*
T0*!
_class
loc:@q_eval/weights*

index_type0*
_output_shapes
:	
Г
q_eval/q_eval/weights/Adam
VariableV2*
shared_name *!
_class
loc:@q_eval/weights*
	container *
shape:	*
dtype0*
_output_shapes
:	
ы
!q_eval/q_eval/weights/Adam/AssignAssignq_eval/q_eval/weights/Adam,q_eval/q_eval/weights/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	

q_eval/q_eval/weights/Adam/readIdentityq_eval/q_eval/weights/Adam*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
:	
В
>q_eval/q_eval/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@q_eval/weights*
valueB"      *
dtype0*
_output_shapes
:

4q_eval/q_eval/weights/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@q_eval/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

.q_eval/q_eval/weights/Adam_1/Initializer/zerosFill>q_eval/q_eval/weights/Adam_1/Initializer/zeros/shape_as_tensor4q_eval/q_eval/weights/Adam_1/Initializer/zeros/Const*
T0*!
_class
loc:@q_eval/weights*

index_type0*
_output_shapes
:	
Е
q_eval/q_eval/weights/Adam_1
VariableV2*
shared_name *!
_class
loc:@q_eval/weights*
	container *
shape:	*
dtype0*
_output_shapes
:	
ё
#q_eval/q_eval/weights/Adam_1/AssignAssignq_eval/q_eval/weights/Adam_1.q_eval/q_eval/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	

!q_eval/q_eval/weights/Adam_1/readIdentityq_eval/q_eval/weights/Adam_1*
T0*!
_class
loc:@q_eval/weights*
_output_shapes
:	

+q_eval/q_eval/biases/Adam/Initializer/zerosConst* 
_class
loc:@q_eval/biases*
valueB*    *
dtype0*
_output_shapes
:
Ї
q_eval/q_eval/biases/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@q_eval/biases*
	container 
т
 q_eval/q_eval/biases/Adam/AssignAssignq_eval/q_eval/biases/Adam+q_eval/q_eval/biases/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:

q_eval/q_eval/biases/Adam/readIdentityq_eval/q_eval/biases/Adam*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
:

-q_eval/q_eval/biases/Adam_1/Initializer/zerosConst* 
_class
loc:@q_eval/biases*
valueB*    *
dtype0*
_output_shapes
:
Љ
q_eval/q_eval/biases/Adam_1
VariableV2* 
_class
loc:@q_eval/biases*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ш
"q_eval/q_eval/biases/Adam_1/AssignAssignq_eval/q_eval/biases/Adam_1-q_eval/q_eval/biases/Adam_1/Initializer/zeros*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:*
use_locking(

 q_eval/q_eval/biases/Adam_1/readIdentityq_eval/q_eval/biases/Adam_1*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
:
^
q_eval/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o9
V
q_eval/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
V
q_eval/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
X
q_eval/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
ф
0q_eval/Adam/update_q_eval/conv1/kernel/ApplyAdam	ApplyAdamq_eval/conv1/kernelq_eval/q_eval/conv1/kernel/Adam!q_eval/q_eval/conv1/kernel/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonDq_eval/gradients/q_eval/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*&
_class
loc:@q_eval/conv1/kernel*
use_nesterov( *&
_output_shapes
:*
use_locking( 
Я
.q_eval/Adam/update_q_eval/conv1/bias/ApplyAdam	ApplyAdamq_eval/conv1/biasq_eval/q_eval/conv1/bias/Adamq_eval/q_eval/conv1/bias/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonEq_eval/gradients/q_eval/conv1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*$
_class
loc:@q_eval/conv1/bias
ф
0q_eval/Adam/update_q_eval/conv2/kernel/ApplyAdam	ApplyAdamq_eval/conv2/kernelq_eval/q_eval/conv2/kernel/Adam!q_eval/q_eval/conv2/kernel/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonDq_eval/gradients/q_eval/conv2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@q_eval/conv2/kernel*
use_nesterov( *&
_output_shapes
: 
Я
.q_eval/Adam/update_q_eval/conv2/bias/ApplyAdam	ApplyAdamq_eval/conv2/biasq_eval/q_eval/conv2/bias/Adamq_eval/q_eval/conv2/bias/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonEq_eval/gradients/q_eval/conv2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
: *
use_locking( *
T0*$
_class
loc:@q_eval/conv2/bias

8q_eval/Adam/update_q_eval/rnn/lstm_cell/kernel/ApplyAdam	ApplyAdamq_eval/rnn/lstm_cell/kernel'q_eval/q_eval/rnn/lstm_cell/kernel/Adam)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonEq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
use_nesterov( * 
_output_shapes
:
 
љ
6q_eval/Adam/update_q_eval/rnn/lstm_cell/bias/ApplyAdam	ApplyAdamq_eval/rnn/lstm_cell/bias%q_eval/q_eval/rnn/lstm_cell/bias/Adam'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilonFq_eval/gradients/q_eval/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
О
+q_eval/Adam/update_q_eval/weights/ApplyAdam	ApplyAdamq_eval/weightsq_eval/q_eval/weights/Adamq_eval/q_eval/weights/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilon>q_eval/gradients/q_eval/MatMul_grad/tuple/control_dependency_1*
T0*!
_class
loc:@q_eval/weights*
use_nesterov( *
_output_shapes
:	*
use_locking( 
Б
*q_eval/Adam/update_q_eval/biases/ApplyAdam	ApplyAdamq_eval/biasesq_eval/q_eval/biases/Adamq_eval/q_eval/biases/Adam_1q_eval/beta1_power/readq_eval/beta2_power/readq_eval/Adam/learning_rateq_eval/Adam/beta1q_eval/Adam/beta2q_eval/Adam/epsilon;q_eval/gradients/q_eval/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0* 
_class
loc:@q_eval/biases

q_eval/Adam/mulMulq_eval/beta1_power/readq_eval/Adam/beta1+^q_eval/Adam/update_q_eval/biases/ApplyAdam/^q_eval/Adam/update_q_eval/conv1/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv1/kernel/ApplyAdam/^q_eval/Adam/update_q_eval/conv2/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv2/kernel/ApplyAdam7^q_eval/Adam/update_q_eval/rnn/lstm_cell/bias/ApplyAdam9^q_eval/Adam/update_q_eval/rnn/lstm_cell/kernel/ApplyAdam,^q_eval/Adam/update_q_eval/weights/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@q_eval/biases
­
q_eval/Adam/AssignAssignq_eval/beta1_powerq_eval/Adam/mul*
use_locking( *
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: 

q_eval/Adam/mul_1Mulq_eval/beta2_power/readq_eval/Adam/beta2+^q_eval/Adam/update_q_eval/biases/ApplyAdam/^q_eval/Adam/update_q_eval/conv1/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv1/kernel/ApplyAdam/^q_eval/Adam/update_q_eval/conv2/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv2/kernel/ApplyAdam7^q_eval/Adam/update_q_eval/rnn/lstm_cell/bias/ApplyAdam9^q_eval/Adam/update_q_eval/rnn/lstm_cell/kernel/ApplyAdam,^q_eval/Adam/update_q_eval/weights/ApplyAdam*
T0* 
_class
loc:@q_eval/biases*
_output_shapes
: 
Б
q_eval/Adam/Assign_1Assignq_eval/beta2_powerq_eval/Adam/mul_1*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: *
use_locking( 
ж
q_eval/AdamNoOp^q_eval/Adam/Assign^q_eval/Adam/Assign_1+^q_eval/Adam/update_q_eval/biases/ApplyAdam/^q_eval/Adam/update_q_eval/conv1/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv1/kernel/ApplyAdam/^q_eval/Adam/update_q_eval/conv2/bias/ApplyAdam1^q_eval/Adam/update_q_eval/conv2/kernel/ApplyAdam7^q_eval/Adam/update_q_eval/rnn/lstm_cell/bias/ApplyAdam9^q_eval/Adam/update_q_eval/rnn/lstm_cell/kernel/ApplyAdam,^q_eval/Adam/update_q_eval/weights/ApplyAdam
Ё
2q_eval/q_eval/conv1/kernel/summaries/histogram/tagConst*
dtype0*
_output_shapes
: *?
value6B4 B.q_eval/q_eval/conv1/kernel/summaries/histogram
Б
.q_eval/q_eval/conv1/kernel/summaries/histogramHistogramSummary2q_eval/q_eval/conv1/kernel/summaries/histogram/tagq_eval/conv1/kernel/read*
T0*
_output_shapes
: 

0q_eval/q_eval/conv1/bias/summaries/histogram/tagConst*
dtype0*
_output_shapes
: *=
value4B2 B,q_eval/q_eval/conv1/bias/summaries/histogram
Ћ
,q_eval/q_eval/conv1/bias/summaries/histogramHistogramSummary0q_eval/q_eval/conv1/bias/summaries/histogram/tagq_eval/conv1/bias/read*
_output_shapes
: *
T0
Ё
2q_eval/q_eval/conv2/kernel/summaries/histogram/tagConst*
dtype0*
_output_shapes
: *?
value6B4 B.q_eval/q_eval/conv2/kernel/summaries/histogram
Б
.q_eval/q_eval/conv2/kernel/summaries/histogramHistogramSummary2q_eval/q_eval/conv2/kernel/summaries/histogram/tagq_eval/conv2/kernel/read*
T0*
_output_shapes
: 

0q_eval/q_eval/conv2/bias/summaries/histogram/tagConst*=
value4B2 B,q_eval/q_eval/conv2/bias/summaries/histogram*
dtype0*
_output_shapes
: 
Ћ
,q_eval/q_eval/conv2/bias/summaries/histogramHistogramSummary0q_eval/q_eval/conv2/bias/summaries/histogram/tagq_eval/conv2/bias/read*
_output_shapes
: *
T0
Б
:q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram/tagConst*G
value>B< B6q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram*
dtype0*
_output_shapes
: 
Щ
6q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogramHistogramSummary:q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram/tag q_eval/rnn/lstm_cell/kernel/read*
T0*
_output_shapes
: 
­
8q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram/tagConst*E
value<B: B4q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram*
dtype0*
_output_shapes
: 
У
4q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogramHistogramSummary8q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram/tagq_eval/rnn/lstm_cell/bias/read*
T0*
_output_shapes
: 

-q_eval/q_eval/weights/summaries/histogram/tagConst*:
value1B/ B)q_eval/q_eval/weights/summaries/histogram*
dtype0*
_output_shapes
: 
Ђ
)q_eval/q_eval/weights/summaries/histogramHistogramSummary-q_eval/q_eval/weights/summaries/histogram/tagq_eval/weights/read*
T0*
_output_shapes
: 

,q_eval/q_eval/biases/summaries/histogram/tagConst*9
value0B. B(q_eval/q_eval/biases/summaries/histogram*
dtype0*
_output_shapes
: 

(q_eval/q_eval/biases/summaries/histogramHistogramSummary,q_eval/q_eval/biases/summaries/histogram/tagq_eval/biases/read*
T0*
_output_shapes
: 
Щ
initNoOp^q_eval/beta1_power/Assign^q_eval/beta2_power/Assign^q_eval/biases/Assign^q_eval/conv1/bias/Assign^q_eval/conv1/kernel/Assign^q_eval/conv2/bias/Assign^q_eval/conv2/kernel/Assign!^q_eval/q_eval/biases/Adam/Assign#^q_eval/q_eval/biases/Adam_1/Assign%^q_eval/q_eval/conv1/bias/Adam/Assign'^q_eval/q_eval/conv1/bias/Adam_1/Assign'^q_eval/q_eval/conv1/kernel/Adam/Assign)^q_eval/q_eval/conv1/kernel/Adam_1/Assign%^q_eval/q_eval/conv2/bias/Adam/Assign'^q_eval/q_eval/conv2/bias/Adam_1/Assign'^q_eval/q_eval/conv2/kernel/Adam/Assign)^q_eval/q_eval/conv2/kernel/Adam_1/Assign-^q_eval/q_eval/rnn/lstm_cell/bias/Adam/Assign/^q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Assign/^q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Assign1^q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Assign"^q_eval/q_eval/weights/Adam/Assign$^q_eval/q_eval/weights/Adam_1/Assign!^q_eval/rnn/lstm_cell/bias/Assign#^q_eval/rnn/lstm_cell/kernel/Assign^q_eval/weights/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Я
save/SaveV2/tensor_namesConst*
valueјBѕBq_eval/beta1_powerBq_eval/beta2_powerBq_eval/biasesBq_eval/conv1/biasBq_eval/conv1/kernelBq_eval/conv2/biasBq_eval/conv2/kernelBq_eval/q_eval/biases/AdamBq_eval/q_eval/biases/Adam_1Bq_eval/q_eval/conv1/bias/AdamBq_eval/q_eval/conv1/bias/Adam_1Bq_eval/q_eval/conv1/kernel/AdamB!q_eval/q_eval/conv1/kernel/Adam_1Bq_eval/q_eval/conv2/bias/AdamBq_eval/q_eval/conv2/bias/Adam_1Bq_eval/q_eval/conv2/kernel/AdamB!q_eval/q_eval/conv2/kernel/Adam_1B%q_eval/q_eval/rnn/lstm_cell/bias/AdamB'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1B'q_eval/q_eval/rnn/lstm_cell/kernel/AdamB)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1Bq_eval/q_eval/weights/AdamBq_eval/q_eval/weights/Adam_1Bq_eval/rnn/lstm_cell/biasBq_eval/rnn/lstm_cell/kernelBq_eval/weights*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
№
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesq_eval/beta1_powerq_eval/beta2_powerq_eval/biasesq_eval/conv1/biasq_eval/conv1/kernelq_eval/conv2/biasq_eval/conv2/kernelq_eval/q_eval/biases/Adamq_eval/q_eval/biases/Adam_1q_eval/q_eval/conv1/bias/Adamq_eval/q_eval/conv1/bias/Adam_1q_eval/q_eval/conv1/kernel/Adam!q_eval/q_eval/conv1/kernel/Adam_1q_eval/q_eval/conv2/bias/Adamq_eval/q_eval/conv2/bias/Adam_1q_eval/q_eval/conv2/kernel/Adam!q_eval/q_eval/conv2/kernel/Adam_1%q_eval/q_eval/rnn/lstm_cell/bias/Adam'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1'q_eval/q_eval/rnn/lstm_cell/kernel/Adam)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1q_eval/q_eval/weights/Adamq_eval/q_eval/weights/Adam_1q_eval/rnn/lstm_cell/biasq_eval/rnn/lstm_cell/kernelq_eval/weights*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
с
save/RestoreV2/tensor_namesConst"/device:CPU:0*
valueјBѕBq_eval/beta1_powerBq_eval/beta2_powerBq_eval/biasesBq_eval/conv1/biasBq_eval/conv1/kernelBq_eval/conv2/biasBq_eval/conv2/kernelBq_eval/q_eval/biases/AdamBq_eval/q_eval/biases/Adam_1Bq_eval/q_eval/conv1/bias/AdamBq_eval/q_eval/conv1/bias/Adam_1Bq_eval/q_eval/conv1/kernel/AdamB!q_eval/q_eval/conv1/kernel/Adam_1Bq_eval/q_eval/conv2/bias/AdamBq_eval/q_eval/conv2/bias/Adam_1Bq_eval/q_eval/conv2/kernel/AdamB!q_eval/q_eval/conv2/kernel/Adam_1B%q_eval/q_eval/rnn/lstm_cell/bias/AdamB'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1B'q_eval/q_eval/rnn/lstm_cell/kernel/AdamB)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1Bq_eval/q_eval/weights/AdamBq_eval/q_eval/weights/Adam_1Bq_eval/rnn/lstm_cell/biasBq_eval/rnn/lstm_cell/kernelBq_eval/weights*
dtype0*
_output_shapes
:
Љ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
Ѕ
save/AssignAssignq_eval/beta1_powersave/RestoreV2*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: 
Љ
save/Assign_1Assignq_eval/beta2_powersave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@q_eval/biases
Ј
save/Assign_2Assignq_eval/biasessave/RestoreV2:2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_eval/biases
А
save/Assign_3Assignq_eval/conv1/biassave/RestoreV2:3*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Р
save/Assign_4Assignq_eval/conv1/kernelsave/RestoreV2:4*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel
А
save/Assign_5Assignq_eval/conv2/biassave/RestoreV2:5*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Р
save/Assign_6Assignq_eval/conv2/kernelsave/RestoreV2:6*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(
Д
save/Assign_7Assignq_eval/q_eval/biases/Adamsave/RestoreV2:7*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_eval/biases
Ж
save/Assign_8Assignq_eval/q_eval/biases/Adam_1save/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:
М
save/Assign_9Assignq_eval/q_eval/conv1/bias/Adamsave/RestoreV2:9*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Р
save/Assign_10Assignq_eval/q_eval/conv1/bias/Adam_1save/RestoreV2:10*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:
Ю
save/Assign_11Assignq_eval/q_eval/conv1/kernel/Adamsave/RestoreV2:11*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:
а
save/Assign_12Assign!q_eval/q_eval/conv1/kernel/Adam_1save/RestoreV2:12*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel
О
save/Assign_13Assignq_eval/q_eval/conv2/bias/Adamsave/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: 
Р
save/Assign_14Assignq_eval/q_eval/conv2/bias/Adam_1save/RestoreV2:14*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias
Ю
save/Assign_15Assignq_eval/q_eval/conv2/kernel/Adamsave/RestoreV2:15*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(
а
save/Assign_16Assign!q_eval/q_eval/conv2/kernel/Adam_1save/RestoreV2:16*
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: 
Я
save/Assign_17Assign%q_eval/q_eval/rnn/lstm_cell/bias/Adamsave/RestoreV2:17*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias
б
save/Assign_18Assign'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1save/RestoreV2:18*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
и
save/Assign_19Assign'q_eval/q_eval/rnn/lstm_cell/kernel/Adamsave/RestoreV2:19*
use_locking(*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
к
save/Assign_20Assign)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1save/RestoreV2:20*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
Н
save/Assign_21Assignq_eval/q_eval/weights/Adamsave/RestoreV2:21*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
П
save/Assign_22Assignq_eval/q_eval/weights/Adam_1save/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
У
save/Assign_23Assignq_eval/rnn/lstm_cell/biassave/RestoreV2:23*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
Ь
save/Assign_24Assignq_eval/rnn/lstm_cell/kernelsave/RestoreV2:24*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
Б
save/Assign_25Assignq_eval/weightssave/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
Ц
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
О
Merge/MergeSummaryMergeSummaryq_eval/Reward/Time_step_1#q_eval/TotalWaitingTime/Time_step_1q_eval/TotalDelay/Time_step_1q_eval/Q_valueq_eval/Loss.q_eval/q_eval/conv1/kernel/summaries/histogram,q_eval/q_eval/conv1/bias/summaries/histogram.q_eval/q_eval/conv2/kernel/summaries/histogram,q_eval/q_eval/conv2/bias/summaries/histogram6q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram4q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram)q_eval/q_eval/weights/summaries/histogram(q_eval/q_eval/biases/summaries/histogram*
N*
_output_shapes
: 

q_next/statesPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџTT*$
shape:џџџџџџџџџTT
v
q_next/action_takenPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
i
q_next/q_valuePlaceholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
[
q_next/sequence_lengthPlaceholder*
shape:*
dtype0*
_output_shapes
:
V
q_next/batch_sizePlaceholder*
shape:*
dtype0*
_output_shapes
:
v
q_next/cell_statePlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
s
q_next/h_statePlaceholder*
shape:џџџџџџџџџ*
dtype0*(
_output_shapes
:џџџџџџџџџ
X
q_next/Reward/Time_stepPlaceholder*
dtype0*
_output_shapes
: *
shape: 
x
q_next/Reward/Time_step_1/tagsConst**
value!B Bq_next/Reward/Time_step_1*
dtype0*
_output_shapes
: 

q_next/Reward/Time_step_1ScalarSummaryq_next/Reward/Time_step_1/tagsq_next/Reward/Time_step*
T0*
_output_shapes
: 
b
!q_next/TotalWaitingTime/Time_stepPlaceholder*
shape: *
dtype0*
_output_shapes
: 

(q_next/TotalWaitingTime/Time_step_1/tagsConst*
dtype0*
_output_shapes
: *4
value+B) B#q_next/TotalWaitingTime/Time_step_1
Ђ
#q_next/TotalWaitingTime/Time_step_1ScalarSummary(q_next/TotalWaitingTime/Time_step_1/tags!q_next/TotalWaitingTime/Time_step*
T0*
_output_shapes
: 
\
q_next/TotalDelay/Time_stepPlaceholder*
shape: *
dtype0*
_output_shapes
: 

"q_next/TotalDelay/Time_step_1/tagsConst*
dtype0*
_output_shapes
: *.
value%B# Bq_next/TotalDelay/Time_step_1

q_next/TotalDelay/Time_step_1ScalarSummary"q_next/TotalDelay/Time_step_1/tagsq_next/TotalDelay/Time_step*
_output_shapes
: *
T0
З
6q_next/conv1/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*&
_class
loc:@q_next/conv1/kernel*%
valueB"            
Ђ
5q_next/conv1/kernel/Initializer/truncated_normal/meanConst*&
_class
loc:@q_next/conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
7q_next/conv1/kernel/Initializer/truncated_normal/stddevConst*&
_class
loc:@q_next/conv1/kernel*
valueB
 *аdN>*
dtype0*
_output_shapes
: 

@q_next/conv1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6q_next/conv1/kernel/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:*

seed *
T0*&
_class
loc:@q_next/conv1/kernel*
seed2 

4q_next/conv1/kernel/Initializer/truncated_normal/mulMul@q_next/conv1/kernel/Initializer/truncated_normal/TruncatedNormal7q_next/conv1/kernel/Initializer/truncated_normal/stddev*&
_output_shapes
:*
T0*&
_class
loc:@q_next/conv1/kernel
§
0q_next/conv1/kernel/Initializer/truncated_normalAdd4q_next/conv1/kernel/Initializer/truncated_normal/mul5q_next/conv1/kernel/Initializer/truncated_normal/mean*
T0*&
_class
loc:@q_next/conv1/kernel*&
_output_shapes
:
П
q_next/conv1/kernel
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *&
_class
loc:@q_next/conv1/kernel*
	container *
shape:
э
q_next/conv1/kernel/AssignAssignq_next/conv1/kernel0q_next/conv1/kernel/Initializer/truncated_normal*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_next/conv1/kernel

q_next/conv1/kernel/readIdentityq_next/conv1/kernel*
T0*&
_class
loc:@q_next/conv1/kernel*&
_output_shapes
:

#q_next/conv1/bias/Initializer/ConstConst*$
_class
loc:@q_next/conv1/bias*
valueB*
з#<*
dtype0*
_output_shapes
:
Ѓ
q_next/conv1/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@q_next/conv1/bias*
	container *
shape:
Ю
q_next/conv1/bias/AssignAssignq_next/conv1/bias#q_next/conv1/bias/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_next/conv1/bias

q_next/conv1/bias/readIdentityq_next/conv1/bias*
T0*$
_class
loc:@q_next/conv1/bias*
_output_shapes
:
k
q_next/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
я
q_next/conv1/Conv2DConv2Dq_next/statesq_next/conv1/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ

q_next/conv1/BiasAddBiasAddq_next/conv1/Conv2Dq_next/conv1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ
c
q_next/ReluReluq_next/conv1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ
З
6q_next/conv2/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*&
_class
loc:@q_next/conv2/kernel*%
valueB"             
Ђ
5q_next/conv2/kernel/Initializer/truncated_normal/meanConst*&
_class
loc:@q_next/conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Є
7q_next/conv2/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_next/conv2/kernel*
valueB
 *аdЮ=

@q_next/conv2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6q_next/conv2/kernel/Initializer/truncated_normal/shape*
T0*&
_class
loc:@q_next/conv2/kernel*
seed2 *
dtype0*&
_output_shapes
: *

seed 

4q_next/conv2/kernel/Initializer/truncated_normal/mulMul@q_next/conv2/kernel/Initializer/truncated_normal/TruncatedNormal7q_next/conv2/kernel/Initializer/truncated_normal/stddev*&
_output_shapes
: *
T0*&
_class
loc:@q_next/conv2/kernel
§
0q_next/conv2/kernel/Initializer/truncated_normalAdd4q_next/conv2/kernel/Initializer/truncated_normal/mul5q_next/conv2/kernel/Initializer/truncated_normal/mean*
T0*&
_class
loc:@q_next/conv2/kernel*&
_output_shapes
: 
П
q_next/conv2/kernel
VariableV2*&
_class
loc:@q_next/conv2/kernel*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name 
э
q_next/conv2/kernel/AssignAssignq_next/conv2/kernel0q_next/conv2/kernel/Initializer/truncated_normal*
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel*
validate_shape(*&
_output_shapes
: 

q_next/conv2/kernel/readIdentityq_next/conv2/kernel*
T0*&
_class
loc:@q_next/conv2/kernel*&
_output_shapes
: 

#q_next/conv2/bias/Initializer/ConstConst*
dtype0*
_output_shapes
: *$
_class
loc:@q_next/conv2/bias*
valueB *
з#<
Ѓ
q_next/conv2/bias
VariableV2*
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@q_next/conv2/bias*
	container *
shape: 
Ю
q_next/conv2/bias/AssignAssignq_next/conv2/bias#q_next/conv2/bias/Initializer/Const*
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias*
validate_shape(*
_output_shapes
: 

q_next/conv2/bias/readIdentityq_next/conv2/bias*
T0*$
_class
loc:@q_next/conv2/bias*
_output_shapes
: 
k
q_next/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
э
q_next/conv2/Conv2DConv2Dq_next/Reluq_next/conv2/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ		 

q_next/conv2/BiasAddBiasAddq_next/conv2/Conv2Dq_next/conv2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ		 
e
q_next/Relu_1Reluq_next/conv2/BiasAdd*/
_output_shapes
:џџџџџџџџџ		 *
T0
e
q_next/Reshape/shapeConst*
valueB"џџџџ 
  *
dtype0*
_output_shapes
:

q_next/ReshapeReshapeq_next/Relu_1q_next/Reshape/shape*(
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0
[
q_next/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value
B : 

q_next/Reshape_1/shapePackq_next/batch_sizeq_next/sequence_lengthq_next/Reshape_1/shape/2*
T0*

axis *
N*
_output_shapes
:

q_next/Reshape_1Reshapeq_next/Reshapeq_next/Reshape_1/shape*
T0*
Tshape0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ 
Q
q_next/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
X
q_next/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
X
q_next/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_next/rnn/rangeRangeq_next/rnn/range/startq_next/rnn/Rankq_next/rnn/range/delta*
_output_shapes
:*

Tidx0
k
q_next/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
X
q_next/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

q_next/rnn/concatConcatV2q_next/rnn/concat/values_0q_next/rnn/rangeq_next/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:

q_next/rnn/transpose	Transposeq_next/Reshape_1q_next/rnn/concat*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
Tperm0
a
q_next/rnn/sequence_lengthIdentityq_next/sequence_length*
_output_shapes
:*
T0
d
q_next/rnn/ShapeShapeq_next/rnn/transpose*
T0*
out_type0*
_output_shapes
:
h
q_next/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
j
 q_next/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
j
 q_next/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
А
q_next/rnn/strided_sliceStridedSliceq_next/rnn/Shapeq_next/rnn/strided_slice/stack q_next/rnn/strided_slice/stack_1 q_next/rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
u
q_next/rnn/Shape_1Shapeq_next/rnn/sequence_length*#
_output_shapes
:џџџџџџџџџ*
T0*
out_type0
l
q_next/rnn/stackPackq_next/rnn/strided_slice*
T0*

axis *
N*
_output_shapes
:
m
q_next/rnn/EqualEqualq_next/rnn/Shape_1q_next/rnn/stack*
T0*#
_output_shapes
:џџџџџџџџџ
Z
q_next/rnn/ConstConst*
valueB: *
dtype0*
_output_shapes
:
n
q_next/rnn/AllAllq_next/rnn/Equalq_next/rnn/Const*
_output_shapes
: *
	keep_dims( *

Tidx0

q_next/rnn/Assert/ConstConst*K
valueBB@ B:Expected shape for Tensor q_next/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
j
q_next/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

q_next/rnn/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *K
valueBB@ B:Expected shape for Tensor q_next/rnn/sequence_length:0 is 
p
q_next/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
К
q_next/rnn/Assert/AssertAssertq_next/rnn/Allq_next/rnn/Assert/Assert/data_0q_next/rnn/stackq_next/rnn/Assert/Assert/data_2q_next/rnn/Shape_1*
T
2*
	summarize
|
q_next/rnn/CheckSeqLenIdentityq_next/rnn/sequence_length^q_next/rnn/Assert/Assert*
T0*
_output_shapes
:
f
q_next/rnn/Shape_2Shapeq_next/rnn/transpose*
T0*
out_type0*
_output_shapes
:
j
 q_next/rnn/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: 
l
"q_next/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"q_next/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
К
q_next/rnn/strided_slice_1StridedSliceq_next/rnn/Shape_2 q_next/rnn/strided_slice_1/stack"q_next/rnn/strided_slice_1/stack_1"q_next/rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
q_next/rnn/Shape_3Shapeq_next/rnn/transpose*
T0*
out_type0*
_output_shapes
:
j
 q_next/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
l
"q_next/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"q_next/rnn/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
К
q_next/rnn/strided_slice_2StridedSliceq_next/rnn/Shape_3 q_next/rnn/strided_slice_2/stack"q_next/rnn/strided_slice_2/stack_1"q_next/rnn/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
[
q_next/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 

q_next/rnn/ExpandDims
ExpandDimsq_next/rnn/strided_slice_2q_next/rnn/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
]
q_next/rnn/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
Z
q_next/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

q_next/rnn/concat_1ConcatV2q_next/rnn/ExpandDimsq_next/rnn/Const_1q_next/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
[
q_next/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

q_next/rnn/zerosFillq_next/rnn/concat_1q_next/rnn/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
R
q_next/rnn/Rank_1Rankq_next/rnn/CheckSeqLen*
_output_shapes
: *
T0
Z
q_next/rnn/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
Z
q_next/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_next/rnn/range_1Rangeq_next/rnn/range_1/startq_next/rnn/Rank_1q_next/rnn/range_1/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ

q_next/rnn/MinMinq_next/rnn/CheckSeqLenq_next/rnn/range_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
q_next/rnn/Rank_2Rankq_next/rnn/CheckSeqLen*
T0*
_output_shapes
: 
Z
q_next/rnn/range_2/startConst*
value	B : *
dtype0*
_output_shapes
: 
Z
q_next/rnn/range_2/deltaConst*
dtype0*
_output_shapes
: *
value	B :

q_next/rnn/range_2Rangeq_next/rnn/range_2/startq_next/rnn/Rank_2q_next/rnn/range_2/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0

q_next/rnn/MaxMaxq_next/rnn/CheckSeqLenq_next/rnn/range_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Q
q_next/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 

q_next/rnn/TensorArrayTensorArrayV3q_next/rnn/strided_slice_1*6
tensor_array_name!q_next/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *%
element_shape:џџџџџџџџџ*
dynamic_size( *
clear_after_read(*
identical_element_shapes(

q_next/rnn/TensorArray_1TensorArrayV3q_next/rnn/strided_slice_1*%
element_shape:џџџџџџџџџ *
dynamic_size( *
clear_after_read(*
identical_element_shapes(*5
tensor_array_name q_next/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
w
#q_next/rnn/TensorArrayUnstack/ShapeShapeq_next/rnn/transpose*
T0*
out_type0*
_output_shapes
:
{
1q_next/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
}
3q_next/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
}
3q_next/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

+q_next/rnn/TensorArrayUnstack/strided_sliceStridedSlice#q_next/rnn/TensorArrayUnstack/Shape1q_next/rnn/TensorArrayUnstack/strided_slice/stack3q_next/rnn/TensorArrayUnstack/strided_slice/stack_13q_next/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
k
)q_next/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
k
)q_next/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
р
#q_next/rnn/TensorArrayUnstack/rangeRange)q_next/rnn/TensorArrayUnstack/range/start+q_next/rnn/TensorArrayUnstack/strided_slice)q_next/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0

Eq_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3q_next/rnn/TensorArray_1#q_next/rnn/TensorArrayUnstack/rangeq_next/rnn/transposeq_next/rnn/TensorArray_1:1*
_output_shapes
: *
T0*'
_class
loc:@q_next/rnn/transpose
V
q_next/rnn/Maximum/xConst*
dtype0*
_output_shapes
: *
value	B :
d
q_next/rnn/MaximumMaximumq_next/rnn/Maximum/xq_next/rnn/Max*
T0*
_output_shapes
: 
n
q_next/rnn/MinimumMinimumq_next/rnn/strided_slice_1q_next/rnn/Maximum*
T0*
_output_shapes
: 
d
"q_next/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Т
q_next/rnn/while/EnterEnter"q_next/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context
Б
q_next/rnn/while/Enter_1Enterq_next/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context
К
q_next/rnn/while/Enter_2Enterq_next/rnn/TensorArray:1*
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context*
T0*
is_constant( 
Х
q_next/rnn/while/Enter_3Enterq_next/cell_state*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*.

frame_name q_next/rnn/while/while_context
Т
q_next/rnn/while/Enter_4Enterq_next/h_state*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*.

frame_name q_next/rnn/while/while_context

q_next/rnn/while/MergeMergeq_next/rnn/while/Enterq_next/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 

q_next/rnn/while/Merge_1Mergeq_next/rnn/while/Enter_1 q_next/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 

q_next/rnn/while/Merge_2Mergeq_next/rnn/while/Enter_2 q_next/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

q_next/rnn/while/Merge_3Mergeq_next/rnn/while/Enter_3 q_next/rnn/while/NextIteration_3*
N**
_output_shapes
:џџџџџџџџџ: *
T0

q_next/rnn/while/Merge_4Mergeq_next/rnn/while/Enter_4 q_next/rnn/while/NextIteration_4*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
s
q_next/rnn/while/LessLessq_next/rnn/while/Mergeq_next/rnn/while/Less/Enter*
T0*
_output_shapes
: 
П
q_next/rnn/while/Less/EnterEnterq_next/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context
y
q_next/rnn/while/Less_1Lessq_next/rnn/while/Merge_1q_next/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
Й
q_next/rnn/while/Less_1/EnterEnterq_next/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context
q
q_next/rnn/while/LogicalAnd
LogicalAndq_next/rnn/while/Lessq_next/rnn/while/Less_1*
_output_shapes
: 
Z
q_next/rnn/while/LoopCondLoopCondq_next/rnn/while/LogicalAnd*
_output_shapes
: 
Ђ
q_next/rnn/while/SwitchSwitchq_next/rnn/while/Mergeq_next/rnn/while/LoopCond*
_output_shapes
: : *
T0*)
_class
loc:@q_next/rnn/while/Merge
Ј
q_next/rnn/while/Switch_1Switchq_next/rnn/while/Merge_1q_next/rnn/while/LoopCond*
_output_shapes
: : *
T0*+
_class!
loc:@q_next/rnn/while/Merge_1
Ј
q_next/rnn/while/Switch_2Switchq_next/rnn/while/Merge_2q_next/rnn/while/LoopCond*
T0*+
_class!
loc:@q_next/rnn/while/Merge_2*
_output_shapes
: : 
Ь
q_next/rnn/while/Switch_3Switchq_next/rnn/while/Merge_3q_next/rnn/while/LoopCond*
T0*+
_class!
loc:@q_next/rnn/while/Merge_3*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
Ь
q_next/rnn/while/Switch_4Switchq_next/rnn/while/Merge_4q_next/rnn/while/LoopCond*
T0*+
_class!
loc:@q_next/rnn/while/Merge_4*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
a
q_next/rnn/while/IdentityIdentityq_next/rnn/while/Switch:1*
_output_shapes
: *
T0
e
q_next/rnn/while/Identity_1Identityq_next/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
e
q_next/rnn/while/Identity_2Identityq_next/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
w
q_next/rnn/while/Identity_3Identityq_next/rnn/while/Switch_3:1*(
_output_shapes
:џџџџџџџџџ*
T0
w
q_next/rnn/while/Identity_4Identityq_next/rnn/while/Switch_4:1*
T0*(
_output_shapes
:џџџџџџџџџ
t
q_next/rnn/while/add/yConst^q_next/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
o
q_next/rnn/while/addAddq_next/rnn/while/Identityq_next/rnn/while/add/y*
T0*
_output_shapes
: 
с
"q_next/rnn/while/TensorArrayReadV3TensorArrayReadV3(q_next/rnn/while/TensorArrayReadV3/Enterq_next/rnn/while/Identity_1*q_next/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:џџџџџџџџџ 
Ю
(q_next/rnn/while/TensorArrayReadV3/EnterEnterq_next/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
љ
*q_next/rnn/while/TensorArrayReadV3/Enter_1EnterEq_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context

q_next/rnn/while/GreaterEqualGreaterEqualq_next/rnn/while/Identity_1#q_next/rnn/while/GreaterEqual/Enter*
_output_shapes
:*
T0
Х
#q_next/rnn/while/GreaterEqual/EnterEnterq_next/rnn/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Н
<q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Џ
:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB
 *њ<!Н*
dtype0*
_output_shapes
: 
Џ
:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB
 *њ<!=*
dtype0*
_output_shapes
: 

Dq_next/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform<q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
seed2 *
dtype0* 
_output_shapes
:
 *

seed 

:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/subSub:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/max:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
_output_shapes
: 

:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMulDq_next/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniform:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel* 
_output_shapes
:
 

6q_next/rnn/lstm_cell/kernel/Initializer/random_uniformAdd:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/mul:q_next/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel* 
_output_shapes
:
 
У
q_next/rnn/lstm_cell/kernel
VariableV2*
shared_name *.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
	container *
shape:
 *
dtype0* 
_output_shapes
:
 

"q_next/rnn/lstm_cell/kernel/AssignAssignq_next/rnn/lstm_cell/kernel6q_next/rnn/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
t
 q_next/rnn/lstm_cell/kernel/readIdentityq_next/rnn/lstm_cell/kernel*
T0* 
_output_shapes
:
 
Д
;q_next/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
Є
1q_next/rnn/lstm_cell/bias/Initializer/zeros/ConstConst*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

+q_next/rnn/lstm_cell/bias/Initializer/zerosFill;q_next/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensor1q_next/rnn/lstm_cell/bias/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
Е
q_next/rnn/lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
	container *
shape:
я
 q_next/rnn/lstm_cell/bias/AssignAssignq_next/rnn/lstm_cell/bias+q_next/rnn/lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
k
q_next/rnn/lstm_cell/bias/readIdentityq_next/rnn/lstm_cell/bias*
T0*
_output_shapes	
:

&q_next/rnn/while/lstm_cell/concat/axisConst^q_next/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
о
!q_next/rnn/while/lstm_cell/concatConcatV2"q_next/rnn/while/TensorArrayReadV3q_next/rnn/while/Identity_4&q_next/rnn/while/lstm_cell/concat/axis*
T0*
N*(
_output_shapes
:џџџџџџџџџ *

Tidx0
а
!q_next/rnn/while/lstm_cell/MatMulMatMul!q_next/rnn/while/lstm_cell/concat'q_next/rnn/while/lstm_cell/MatMul/Enter*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
л
'q_next/rnn/while/lstm_cell/MatMul/EnterEnter q_next/rnn/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
 *.

frame_name q_next/rnn/while/while_context
Ф
"q_next/rnn/while/lstm_cell/BiasAddBiasAdd!q_next/rnn/while/lstm_cell/MatMul(q_next/rnn/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
е
(q_next/rnn/while/lstm_cell/BiasAdd/EnterEnterq_next/rnn/lstm_cell/bias/read*
parallel_iterations *
_output_shapes	
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(
~
 q_next/rnn/while/lstm_cell/ConstConst^q_next/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

*q_next/rnn/while/lstm_cell/split/split_dimConst^q_next/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
љ
 q_next/rnn/while/lstm_cell/splitSplit*q_next/rnn/while/lstm_cell/split/split_dim"q_next/rnn/while/lstm_cell/BiasAdd*
T0*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split

 q_next/rnn/while/lstm_cell/add/yConst^q_next/rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

q_next/rnn/while/lstm_cell/addAdd"q_next/rnn/while/lstm_cell/split:2 q_next/rnn/while/lstm_cell/add/y*(
_output_shapes
:џџџџџџџџџ*
T0

"q_next/rnn/while/lstm_cell/SigmoidSigmoidq_next/rnn/while/lstm_cell/add*
T0*(
_output_shapes
:џџџџџџџџџ

q_next/rnn/while/lstm_cell/mulMul"q_next/rnn/while/lstm_cell/Sigmoidq_next/rnn/while/Identity_3*
T0*(
_output_shapes
:џџџџџџџџџ

$q_next/rnn/while/lstm_cell/Sigmoid_1Sigmoid q_next/rnn/while/lstm_cell/split*
T0*(
_output_shapes
:џџџџџџџџџ
~
q_next/rnn/while/lstm_cell/TanhTanh"q_next/rnn/while/lstm_cell/split:1*
T0*(
_output_shapes
:џџџџџџџџџ
Ё
 q_next/rnn/while/lstm_cell/mul_1Mul$q_next/rnn/while/lstm_cell/Sigmoid_1q_next/rnn/while/lstm_cell/Tanh*
T0*(
_output_shapes
:џџџџџџџџџ

 q_next/rnn/while/lstm_cell/add_1Addq_next/rnn/while/lstm_cell/mul q_next/rnn/while/lstm_cell/mul_1*
T0*(
_output_shapes
:џџџџџџџџџ

$q_next/rnn/while/lstm_cell/Sigmoid_2Sigmoid"q_next/rnn/while/lstm_cell/split:3*
T0*(
_output_shapes
:џџџџџџџџџ
~
!q_next/rnn/while/lstm_cell/Tanh_1Tanh q_next/rnn/while/lstm_cell/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ѓ
 q_next/rnn/while/lstm_cell/mul_2Mul$q_next/rnn/while/lstm_cell/Sigmoid_2!q_next/rnn/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:џџџџџџџџџ
щ
q_next/rnn/while/SelectSelectq_next/rnn/while/GreaterEqualq_next/rnn/while/Select/Enter q_next/rnn/while/lstm_cell/mul_2*(
_output_shapes
:џџџџџџџџџ*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2
ў
q_next/rnn/while/Select/EnterEnterq_next/rnn/zeros*
is_constant(*(
_output_shapes
:џџџџџџџџџ*.

frame_name q_next/rnn/while/while_context*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*
parallel_iterations 
щ
q_next/rnn/while/Select_1Selectq_next/rnn/while/GreaterEqualq_next/rnn/while/Identity_3 q_next/rnn/while/lstm_cell/add_1*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/add_1*(
_output_shapes
:џџџџџџџџџ
щ
q_next/rnn/while/Select_2Selectq_next/rnn/while/GreaterEqualq_next/rnn/while/Identity_4 q_next/rnn/while/lstm_cell/mul_2*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*(
_output_shapes
:џџџџџџџџџ
Џ
4q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3:q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterq_next/rnn/while/Identity_1q_next/rnn/while/Selectq_next/rnn/while/Identity_2*
_output_shapes
: *
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2

:q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterq_next/rnn/TensorArray*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
v
q_next/rnn/while/add_1/yConst^q_next/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
q_next/rnn/while/add_1Addq_next/rnn/while/Identity_1q_next/rnn/while/add_1/y*
_output_shapes
: *
T0
f
q_next/rnn/while/NextIterationNextIterationq_next/rnn/while/add*
T0*
_output_shapes
: 
j
 q_next/rnn/while/NextIteration_1NextIterationq_next/rnn/while/add_1*
T0*
_output_shapes
: 

 q_next/rnn/while/NextIteration_2NextIteration4q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0

 q_next/rnn/while/NextIteration_3NextIterationq_next/rnn/while/Select_1*(
_output_shapes
:џџџџџџџџџ*
T0

 q_next/rnn/while/NextIteration_4NextIterationq_next/rnn/while/Select_2*
T0*(
_output_shapes
:џџџџџџџџџ
W
q_next/rnn/while/ExitExitq_next/rnn/while/Switch*
T0*
_output_shapes
: 
[
q_next/rnn/while/Exit_1Exitq_next/rnn/while/Switch_1*
_output_shapes
: *
T0
[
q_next/rnn/while/Exit_2Exitq_next/rnn/while/Switch_2*
_output_shapes
: *
T0
m
q_next/rnn/while/Exit_3Exitq_next/rnn/while/Switch_3*
T0*(
_output_shapes
:џџџџџџџџџ
m
q_next/rnn/while/Exit_4Exitq_next/rnn/while/Switch_4*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
-q_next/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3q_next/rnn/TensorArrayq_next/rnn/while/Exit_2*
_output_shapes
: *)
_class
loc:@q_next/rnn/TensorArray

'q_next/rnn/TensorArrayStack/range/startConst*)
_class
loc:@q_next/rnn/TensorArray*
value	B : *
dtype0*
_output_shapes
: 

'q_next/rnn/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *)
_class
loc:@q_next/rnn/TensorArray*
value	B :

!q_next/rnn/TensorArrayStack/rangeRange'q_next/rnn/TensorArrayStack/range/start-q_next/rnn/TensorArrayStack/TensorArraySizeV3'q_next/rnn/TensorArrayStack/range/delta*)
_class
loc:@q_next/rnn/TensorArray*#
_output_shapes
:џџџџџџџџџ*

Tidx0
А
/q_next/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3q_next/rnn/TensorArray!q_next/rnn/TensorArrayStack/rangeq_next/rnn/while/Exit_2*%
element_shape:џџџџџџџџџ*)
_class
loc:@q_next/rnn/TensorArray*
dtype0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
]
q_next/rnn/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
S
q_next/rnn/Rank_3Const*
value	B :*
dtype0*
_output_shapes
: 
Z
q_next/rnn/range_3/startConst*
value	B :*
dtype0*
_output_shapes
: 
Z
q_next/rnn/range_3/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

q_next/rnn/range_3Rangeq_next/rnn/range_3/startq_next/rnn/Rank_3q_next/rnn/range_3/delta*

Tidx0*
_output_shapes
:
m
q_next/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Z
q_next/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
q_next/rnn/concat_2ConcatV2q_next/rnn/concat_2/values_0q_next/rnn/range_3q_next/rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ж
q_next/rnn/transpose_1	Transpose/q_next/rnn/TensorArrayStack/TensorArrayGatherV3q_next/rnn/concat_2*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
Tperm0
Ѓ
/q_next/weights/Initializer/random_uniform/shapeConst*!
_class
loc:@q_next/weights*
valueB"      *
dtype0*
_output_shapes
:

-q_next/weights/Initializer/random_uniform/minConst*!
_class
loc:@q_next/weights*
valueB
 *О*
dtype0*
_output_shapes
: 

-q_next/weights/Initializer/random_uniform/maxConst*!
_class
loc:@q_next/weights*
valueB
 *>*
dtype0*
_output_shapes
: 
ь
7q_next/weights/Initializer/random_uniform/RandomUniformRandomUniform/q_next/weights/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*!
_class
loc:@q_next/weights*
seed2 
ж
-q_next/weights/Initializer/random_uniform/subSub-q_next/weights/Initializer/random_uniform/max-q_next/weights/Initializer/random_uniform/min*
T0*!
_class
loc:@q_next/weights*
_output_shapes
: 
щ
-q_next/weights/Initializer/random_uniform/mulMul7q_next/weights/Initializer/random_uniform/RandomUniform-q_next/weights/Initializer/random_uniform/sub*
T0*!
_class
loc:@q_next/weights*
_output_shapes
:	
л
)q_next/weights/Initializer/random_uniformAdd-q_next/weights/Initializer/random_uniform/mul-q_next/weights/Initializer/random_uniform/min*
_output_shapes
:	*
T0*!
_class
loc:@q_next/weights
Ї
q_next/weights
VariableV2*
shared_name *!
_class
loc:@q_next/weights*
	container *
shape:	*
dtype0*
_output_shapes
:	
а
q_next/weights/AssignAssignq_next/weights)q_next/weights/Initializer/random_uniform*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	*
use_locking(
|
q_next/weights/readIdentityq_next/weights*
_output_shapes
:	*
T0*!
_class
loc:@q_next/weights

/q_next/weights/Regularizer/l2_regularizer/scaleConst*!
_class
loc:@q_next/weights*
valueB
 *
з#<*
dtype0*
_output_shapes
: 

0q_next/weights/Regularizer/l2_regularizer/L2LossL2Lossq_next/weights/read*
T0*!
_class
loc:@q_next/weights*
_output_shapes
: 
з
)q_next/weights/Regularizer/l2_regularizerMul/q_next/weights/Regularizer/l2_regularizer/scale0q_next/weights/Regularizer/l2_regularizer/L2Loss*
T0*!
_class
loc:@q_next/weights*
_output_shapes
: 

q_next/biases/Initializer/ConstConst* 
_class
loc:@q_next/biases*
valueB*
з#<*
dtype0*
_output_shapes
:

q_next/biases
VariableV2* 
_class
loc:@q_next/biases*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
О
q_next/biases/AssignAssignq_next/biasesq_next/biases/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_next/biases
t
q_next/biases/readIdentityq_next/biases*
T0* 
_class
loc:@q_next/biases*
_output_shapes
:
o
q_next/strided_slice/stackConst*!
valueB"    џџџџ    *
dtype0*
_output_shapes
:
q
q_next/strided_slice/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
q
q_next/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
И
q_next/strided_sliceStridedSliceq_next/rnn/transpose_1q_next/strided_slice/stackq_next/strided_slice/stack_1q_next/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*(
_output_shapes
:џџџџџџџџџ*
Index0*
T0

q_next/MatMulMatMulq_next/strided_sliceq_next/weights/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
f

q_next/addAddq_next/MatMulq_next/biases/read*'
_output_shapes
:џџџџџџџџџ*
T0
a
q_next/Q_value/tagConst*
valueB Bq_next/Q_value*
dtype0*
_output_shapes
: 
c
q_next/Q_valueHistogramSummaryq_next/Q_value/tag
q_next/add*
T0*
_output_shapes
: 
d

q_next/MulMul
q_next/addq_next/action_taken*
T0*'
_output_shapes
:џџџџџџџџџ
^
q_next/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 


q_next/SumSum
q_next/Mulq_next/Sum/reduction_indices*
T0*#
_output_shapes
:џџџџџџџџџ*
	keep_dims( *

Tidx0
[

q_next/subSubq_next/q_value
q_next/Sum*#
_output_shapes
:џџџџџџџџџ*
T0
Q
q_next/SquareSquare
q_next/sub*#
_output_shapes
:џџџџџџџџџ*
T0
V
q_next/ConstConst*
valueB: *
dtype0*
_output_shapes
:
n
q_next/MeanMeanq_next/Squareq_next/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
\
q_next/Loss/tagsConst*
valueB Bq_next/Loss*
dtype0*
_output_shapes
: 
\
q_next/LossScalarSummaryq_next/Loss/tagsq_next/Mean*
T0*
_output_shapes
: 
Y
q_next/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
q_next/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?

q_next/gradients/FillFillq_next/gradients/Shapeq_next/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Z
q_next/gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
М
q_next/gradients/f_count_1Enterq_next/gradients/f_count*
parallel_iterations *
_output_shapes
: *.

frame_name q_next/rnn/while/while_context*
T0*
is_constant( 

q_next/gradients/MergeMergeq_next/gradients/f_count_1q_next/gradients/NextIteration*
T0*
N*
_output_shapes
: : 
w
q_next/gradients/SwitchSwitchq_next/gradients/Mergeq_next/rnn/while/LoopCond*
T0*
_output_shapes
: : 
t
q_next/gradients/Add/yConst^q_next/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
q_next/gradients/AddAddq_next/gradients/Switch:1q_next/gradients/Add/y*
T0*
_output_shapes
: 
ъ
q_next/gradients/NextIterationNextIterationq_next/gradients/AddC^q_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPushV2G^q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPushV2G^q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPushV2i^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2M^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2Y^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2[^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1W^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2K^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2Y^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2[^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1G^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2I^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2Y^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2[^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1G^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2I^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2W^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2Y^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1G^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
_output_shapes
: *
T0
\
q_next/gradients/f_count_2Exitq_next/gradients/Switch*
_output_shapes
: *
T0
Z
q_next/gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
Я
q_next/gradients/b_count_1Enterq_next/gradients/f_count_2*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

q_next/gradients/Merge_1Mergeq_next/gradients/b_count_1 q_next/gradients/NextIteration_1*
N*
_output_shapes
: : *
T0

q_next/gradients/GreaterEqualGreaterEqualq_next/gradients/Merge_1#q_next/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
ж
#q_next/gradients/GreaterEqual/EnterEnterq_next/gradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
]
q_next/gradients/b_count_2LoopCondq_next/gradients/GreaterEqual*
_output_shapes
: 
|
q_next/gradients/Switch_1Switchq_next/gradients/Merge_1q_next/gradients/b_count_2*
T0*
_output_shapes
: : 
~
q_next/gradients/SubSubq_next/gradients/Switch_1:1#q_next/gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
Ю
 q_next/gradients/NextIteration_1NextIterationq_next/gradients/Subd^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
^
q_next/gradients/b_count_3Exitq_next/gradients/Switch_1*
_output_shapes
: *
T0
y
/q_next/gradients/q_next/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Џ
)q_next/gradients/q_next/Mean_grad/ReshapeReshapeq_next/gradients/Fill/q_next/gradients/q_next/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
t
'q_next/gradients/q_next/Mean_grad/ShapeShapeq_next/Square*
_output_shapes
:*
T0*
out_type0
Т
&q_next/gradients/q_next/Mean_grad/TileTile)q_next/gradients/q_next/Mean_grad/Reshape'q_next/gradients/q_next/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
v
)q_next/gradients/q_next/Mean_grad/Shape_1Shapeq_next/Square*
_output_shapes
:*
T0*
out_type0
l
)q_next/gradients/q_next/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
q
'q_next/gradients/q_next/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Р
&q_next/gradients/q_next/Mean_grad/ProdProd)q_next/gradients/q_next/Mean_grad/Shape_1'q_next/gradients/q_next/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
s
)q_next/gradients/q_next/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ф
(q_next/gradients/q_next/Mean_grad/Prod_1Prod)q_next/gradients/q_next/Mean_grad/Shape_2)q_next/gradients/q_next/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
m
+q_next/gradients/q_next/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ќ
)q_next/gradients/q_next/Mean_grad/MaximumMaximum(q_next/gradients/q_next/Mean_grad/Prod_1+q_next/gradients/q_next/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
Њ
*q_next/gradients/q_next/Mean_grad/floordivFloorDiv&q_next/gradients/q_next/Mean_grad/Prod)q_next/gradients/q_next/Mean_grad/Maximum*
T0*
_output_shapes
: 

&q_next/gradients/q_next/Mean_grad/CastCast*q_next/gradients/q_next/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
В
)q_next/gradients/q_next/Mean_grad/truedivRealDiv&q_next/gradients/q_next/Mean_grad/Tile&q_next/gradients/q_next/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ

)q_next/gradients/q_next/Square_grad/ConstConst*^q_next/gradients/q_next/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 

'q_next/gradients/q_next/Square_grad/MulMul
q_next/sub)q_next/gradients/q_next/Square_grad/Const*#
_output_shapes
:џџџџџџџџџ*
T0
В
)q_next/gradients/q_next/Square_grad/Mul_1Mul)q_next/gradients/q_next/Mean_grad/truediv'q_next/gradients/q_next/Square_grad/Mul*
T0*#
_output_shapes
:џџџџџџџџџ
t
&q_next/gradients/q_next/sub_grad/ShapeShapeq_next/q_value*
_output_shapes
:*
T0*
out_type0
r
(q_next/gradients/q_next/sub_grad/Shape_1Shape
q_next/Sum*
T0*
out_type0*
_output_shapes
:
о
6q_next/gradients/q_next/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&q_next/gradients/q_next/sub_grad/Shape(q_next/gradients/q_next/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ю
$q_next/gradients/q_next/sub_grad/SumSum)q_next/gradients/q_next/Square_grad/Mul_16q_next/gradients/q_next/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Н
(q_next/gradients/q_next/sub_grad/ReshapeReshape$q_next/gradients/q_next/sub_grad/Sum&q_next/gradients/q_next/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
в
&q_next/gradients/q_next/sub_grad/Sum_1Sum)q_next/gradients/q_next/Square_grad/Mul_18q_next/gradients/q_next/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
v
$q_next/gradients/q_next/sub_grad/NegNeg&q_next/gradients/q_next/sub_grad/Sum_1*
T0*
_output_shapes
:
С
*q_next/gradients/q_next/sub_grad/Reshape_1Reshape$q_next/gradients/q_next/sub_grad/Neg(q_next/gradients/q_next/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

1q_next/gradients/q_next/sub_grad/tuple/group_depsNoOp)^q_next/gradients/q_next/sub_grad/Reshape+^q_next/gradients/q_next/sub_grad/Reshape_1

9q_next/gradients/q_next/sub_grad/tuple/control_dependencyIdentity(q_next/gradients/q_next/sub_grad/Reshape2^q_next/gradients/q_next/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_next/gradients/q_next/sub_grad/Reshape*#
_output_shapes
:џџџџџџџџџ

;q_next/gradients/q_next/sub_grad/tuple/control_dependency_1Identity*q_next/gradients/q_next/sub_grad/Reshape_12^q_next/gradients/q_next/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_next/gradients/q_next/sub_grad/Reshape_1*#
_output_shapes
:џџџџџџџџџ
p
&q_next/gradients/q_next/Sum_grad/ShapeShape
q_next/Mul*
T0*
out_type0*
_output_shapes
:
Ђ
%q_next/gradients/q_next/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B :
Ь
$q_next/gradients/q_next/Sum_grad/addAddq_next/Sum/reduction_indices%q_next/gradients/q_next/Sum_grad/Size*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
_output_shapes
: 
й
$q_next/gradients/q_next/Sum_grad/modFloorMod$q_next/gradients/q_next/Sum_grad/add%q_next/gradients/q_next/Sum_grad/Size*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
_output_shapes
: 
І
(q_next/gradients/q_next/Sum_grad/Shape_1Const*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
Љ
,q_next/gradients/q_next/Sum_grad/range/startConst*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Љ
,q_next/gradients/q_next/Sum_grad/range/deltaConst*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

&q_next/gradients/q_next/Sum_grad/rangeRange,q_next/gradients/q_next/Sum_grad/range/start%q_next/gradients/q_next/Sum_grad/Size,q_next/gradients/q_next/Sum_grad/range/delta*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
_output_shapes
:*

Tidx0
Ј
+q_next/gradients/q_next/Sum_grad/Fill/valueConst*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ђ
%q_next/gradients/q_next/Sum_grad/FillFill(q_next/gradients/q_next/Sum_grad/Shape_1+q_next/gradients/q_next/Sum_grad/Fill/value*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*

index_type0*
_output_shapes
: 
Ю
.q_next/gradients/q_next/Sum_grad/DynamicStitchDynamicStitch&q_next/gradients/q_next/Sum_grad/range$q_next/gradients/q_next/Sum_grad/mod&q_next/gradients/q_next/Sum_grad/Shape%q_next/gradients/q_next/Sum_grad/Fill*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
N*#
_output_shapes
:џџџџџџџџџ
Ї
*q_next/gradients/q_next/Sum_grad/Maximum/yConst*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ј
(q_next/gradients/q_next/Sum_grad/MaximumMaximum.q_next/gradients/q_next/Sum_grad/DynamicStitch*q_next/gradients/q_next/Sum_grad/Maximum/y*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape*#
_output_shapes
:џџџџџџџџџ
ч
)q_next/gradients/q_next/Sum_grad/floordivFloorDiv&q_next/gradients/q_next/Sum_grad/Shape(q_next/gradients/q_next/Sum_grad/Maximum*
_output_shapes
:*
T0*9
_class/
-+loc:@q_next/gradients/q_next/Sum_grad/Shape
б
(q_next/gradients/q_next/Sum_grad/ReshapeReshape;q_next/gradients/q_next/sub_grad/tuple/control_dependency_1.q_next/gradients/q_next/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ц
%q_next/gradients/q_next/Sum_grad/TileTile(q_next/gradients/q_next/Sum_grad/Reshape)q_next/gradients/q_next/Sum_grad/floordiv*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
p
&q_next/gradients/q_next/Mul_grad/ShapeShape
q_next/add*
T0*
out_type0*
_output_shapes
:
{
(q_next/gradients/q_next/Mul_grad/Shape_1Shapeq_next/action_taken*
_output_shapes
:*
T0*
out_type0
о
6q_next/gradients/q_next/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs&q_next/gradients/q_next/Mul_grad/Shape(q_next/gradients/q_next/Mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

$q_next/gradients/q_next/Mul_grad/MulMul%q_next/gradients/q_next/Sum_grad/Tileq_next/action_taken*
T0*'
_output_shapes
:џџџџџџџџџ
Щ
$q_next/gradients/q_next/Mul_grad/SumSum$q_next/gradients/q_next/Mul_grad/Mul6q_next/gradients/q_next/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(q_next/gradients/q_next/Mul_grad/ReshapeReshape$q_next/gradients/q_next/Mul_grad/Sum&q_next/gradients/q_next/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

&q_next/gradients/q_next/Mul_grad/Mul_1Mul
q_next/add%q_next/gradients/q_next/Sum_grad/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
Я
&q_next/gradients/q_next/Mul_grad/Sum_1Sum&q_next/gradients/q_next/Mul_grad/Mul_18q_next/gradients/q_next/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
*q_next/gradients/q_next/Mul_grad/Reshape_1Reshape&q_next/gradients/q_next/Mul_grad/Sum_1(q_next/gradients/q_next/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

1q_next/gradients/q_next/Mul_grad/tuple/group_depsNoOp)^q_next/gradients/q_next/Mul_grad/Reshape+^q_next/gradients/q_next/Mul_grad/Reshape_1

9q_next/gradients/q_next/Mul_grad/tuple/control_dependencyIdentity(q_next/gradients/q_next/Mul_grad/Reshape2^q_next/gradients/q_next/Mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*;
_class1
/-loc:@q_next/gradients/q_next/Mul_grad/Reshape

;q_next/gradients/q_next/Mul_grad/tuple/control_dependency_1Identity*q_next/gradients/q_next/Mul_grad/Reshape_12^q_next/gradients/q_next/Mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_next/gradients/q_next/Mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
s
&q_next/gradients/q_next/add_grad/ShapeShapeq_next/MatMul*
T0*
out_type0*
_output_shapes
:
r
(q_next/gradients/q_next/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
о
6q_next/gradients/q_next/add_grad/BroadcastGradientArgsBroadcastGradientArgs&q_next/gradients/q_next/add_grad/Shape(q_next/gradients/q_next/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
о
$q_next/gradients/q_next/add_grad/SumSum9q_next/gradients/q_next/Mul_grad/tuple/control_dependency6q_next/gradients/q_next/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(q_next/gradients/q_next/add_grad/ReshapeReshape$q_next/gradients/q_next/add_grad/Sum&q_next/gradients/q_next/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
т
&q_next/gradients/q_next/add_grad/Sum_1Sum9q_next/gradients/q_next/Mul_grad/tuple/control_dependency8q_next/gradients/q_next/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
К
*q_next/gradients/q_next/add_grad/Reshape_1Reshape&q_next/gradients/q_next/add_grad/Sum_1(q_next/gradients/q_next/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0

1q_next/gradients/q_next/add_grad/tuple/group_depsNoOp)^q_next/gradients/q_next/add_grad/Reshape+^q_next/gradients/q_next/add_grad/Reshape_1

9q_next/gradients/q_next/add_grad/tuple/control_dependencyIdentity(q_next/gradients/q_next/add_grad/Reshape2^q_next/gradients/q_next/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@q_next/gradients/q_next/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;q_next/gradients/q_next/add_grad/tuple/control_dependency_1Identity*q_next/gradients/q_next/add_grad/Reshape_12^q_next/gradients/q_next/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@q_next/gradients/q_next/add_grad/Reshape_1
н
*q_next/gradients/q_next/MatMul_grad/MatMulMatMul9q_next/gradients/q_next/add_grad/tuple/control_dependencyq_next/weights/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
з
,q_next/gradients/q_next/MatMul_grad/MatMul_1MatMulq_next/strided_slice9q_next/gradients/q_next/add_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0

4q_next/gradients/q_next/MatMul_grad/tuple/group_depsNoOp+^q_next/gradients/q_next/MatMul_grad/MatMul-^q_next/gradients/q_next/MatMul_grad/MatMul_1

<q_next/gradients/q_next/MatMul_grad/tuple/control_dependencyIdentity*q_next/gradients/q_next/MatMul_grad/MatMul5^q_next/gradients/q_next/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_next/gradients/q_next/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

>q_next/gradients/q_next/MatMul_grad/tuple/control_dependency_1Identity,q_next/gradients/q_next/MatMul_grad/MatMul_15^q_next/gradients/q_next/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@q_next/gradients/q_next/MatMul_grad/MatMul_1*
_output_shapes
:	

0q_next/gradients/q_next/strided_slice_grad/ShapeShapeq_next/rnn/transpose_1*
T0*
out_type0*
_output_shapes
:
Ш
;q_next/gradients/q_next/strided_slice_grad/StridedSliceGradStridedSliceGrad0q_next/gradients/q_next/strided_slice_grad/Shapeq_next/strided_slice/stackq_next/strided_slice/stack_1q_next/strided_slice/stack_2<q_next/gradients/q_next/MatMul_grad/tuple/control_dependency*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
T0*
Index0

>q_next/gradients/q_next/rnn/transpose_1_grad/InvertPermutationInvertPermutationq_next/rnn/concat_2*
T0*
_output_shapes
:

6q_next/gradients/q_next/rnn/transpose_1_grad/transpose	Transpose;q_next/gradients/q_next/strided_slice_grad/StridedSliceGrad>q_next/gradients/q_next/rnn/transpose_1_grad/InvertPermutation*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
Tperm0

gq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3q_next/rnn/TensorArrayq_next/rnn/while/Exit_2*)
_class
loc:@q_next/rnn/TensorArray*
sourceq_next/gradients*
_output_shapes

:: 
О
cq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityq_next/rnn/while/Exit_2h^q_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*)
_class
loc:@q_next/rnn/TensorArray*
_output_shapes
: 
Я
mq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3gq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3!q_next/rnn/TensorArrayStack/range6q_next/gradients/q_next/rnn/transpose_1_grad/transposecq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
t
q_next/gradients/zeros_like	ZerosLikeq_next/rnn/while/Exit_3*(
_output_shapes
:џџџџџџџџџ*
T0
v
q_next/gradients/zeros_like_1	ZerosLikeq_next/rnn/while/Exit_4*
T0*(
_output_shapes
:џџџџџџџџџ
М
4q_next/gradients/q_next/rnn/while/Exit_2_grad/b_exitEntermq_next/gradients/q_next/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant( 
ќ
4q_next/gradients/q_next/rnn/while/Exit_3_grad/b_exitEnterq_next/gradients/zeros_like*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ў
4q_next/gradients/q_next/rnn/while/Exit_4_grad/b_exitEnterq_next/gradients/zeros_like_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ф
8q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switchMerge4q_next/gradients/q_next/rnn/while/Exit_2_grad/b_exit?q_next/gradients/q_next/rnn/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
і
8q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switchMerge4q_next/gradients/q_next/rnn/while/Exit_3_grad/b_exit?q_next/gradients/q_next/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:џџџџџџџџџ: 
і
8q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switchMerge4q_next/gradients/q_next/rnn/while/Exit_4_grad/b_exit?q_next/gradients/q_next/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:џџџџџџџџџ: 

5q_next/gradients/q_next/rnn/while/Merge_2_grad/SwitchSwitch8q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switchq_next/gradients/b_count_2*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: : 

?q_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/group_depsNoOp6^q_next/gradients/q_next/rnn/while/Merge_2_grad/Switch
К
Gq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity5q_next/gradients/q_next/rnn/while/Merge_2_grad/Switch@^q_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switch*
_output_shapes
: 
О
Iq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity7q_next/gradients/q_next/rnn/while/Merge_2_grad/Switch:1@^q_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/group_deps*
_output_shapes
: *
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switch
Љ
5q_next/gradients/q_next/rnn/while/Merge_3_grad/SwitchSwitch8q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switchq_next/gradients/b_count_2*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switch*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ

?q_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/group_depsNoOp6^q_next/gradients/q_next/rnn/while/Merge_3_grad/Switch
Ь
Gq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity5q_next/gradients/q_next/rnn/while/Merge_3_grad/Switch@^q_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switch
а
Iq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity7q_next/gradients/q_next/rnn/while/Merge_3_grad/Switch:1@^q_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_3_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
Љ
5q_next/gradients/q_next/rnn/while/Merge_4_grad/SwitchSwitch8q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switchq_next/gradients/b_count_2*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switch

?q_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/group_depsNoOp6^q_next/gradients/q_next/rnn/while/Merge_4_grad/Switch
Ь
Gq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity5q_next/gradients/q_next/rnn/while/Merge_4_grad/Switch@^q_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
а
Iq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity7q_next/gradients/q_next/rnn/while/Merge_4_grad/Switch:1@^q_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_4_grad/b_switch*(
_output_shapes
:џџџџџџџџџ
Ѕ
3q_next/gradients/q_next/rnn/while/Enter_2_grad/ExitExitGq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency*
_output_shapes
: *
T0
З
3q_next/gradients/q_next/rnn/while/Enter_3_grad/ExitExitGq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
З
3q_next/gradients/q_next/rnn/while/Enter_4_grad/ExitExitGq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
Б
lq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterIq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*
sourceq_next/gradients*
_output_shapes

:: 
м
rq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterq_next/rnn/TensorArray*
is_constant(*
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*
parallel_iterations 

hq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityIq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1m^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*3
_class)
'%loc:@q_next/rnn/while/lstm_cell/mul_2*
_output_shapes
: 
щ
\q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3lq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3gq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2hq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:џџџџџџџџџ
н
bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*.
_class$
" loc:@q_next/rnn/while/Identity_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Р
bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
_output_shapes
:*
	elem_type0*.
_class$
" loc:@q_next/rnn/while/Identity_1
в
bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterbq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(
У
hq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2bq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterq_next/rnn/while/Identity_1^q_next/gradients/Add*
_output_shapes
: *
swap_memory( *
T0
Є
gq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2mq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^q_next/gradients/Sub*
_output_shapes
: *
	elem_type0
ю
mq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterbq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
х
cq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerB^q_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2F^q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2F^q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2h^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2L^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2X^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2Z^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1V^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2J^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2X^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2Z^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1F^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2H^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2X^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2Z^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1F^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2H^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2V^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2X^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1F^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2

[q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpJ^q_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1]^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Я
cq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentity\q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3\^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*o
_classe
caloc:@q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*(
_output_shapes
:џџџџџџџџџ

eq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityIq_next/gradients/q_next/rnn/while/Merge_2_grad/tuple/control_dependency_1\^q_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
_output_shapes
: *
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Switch_2_grad/b_switch
С
:q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like	ZerosLikeEq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0
Л
@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/ConstConst*.
_class$
" loc:@q_next/rnn/while/Identity_3*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ќ
@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/f_accStackV2@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/Const*.
_class$
" loc:@q_next/rnn/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0

@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/EnterEnter@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context

Fq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/Enterq_next/rnn/while/Identity_3^q_next/gradients/Add*(
_output_shapes
:џџџџџџџџџ*
swap_memory( *
T0
ђ
Eq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
Н
6q_next/gradients/q_next/rnn/while/Select_1_grad/SelectSelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2Iq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependency_1:q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
Й
<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/ConstConst*0
_class&
$"loc:@q_next/rnn/while/GreaterEqual*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
і
<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/f_accStackV2<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/Const*0
_class&
$"loc:@q_next/rnn/while/GreaterEqual*

stack_name *
_output_shapes
:*
	elem_type0


<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/EnterEnter<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
ћ
Bq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPushV2StackPushV2<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/Enterq_next/rnn/while/GreaterEqual^q_next/gradients/Add*
T0
*
_output_shapes
:*
swap_memory( 
к
Aq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2
StackPopV2Gq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0
*
_output_shapes
:
Ђ
Gq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2/EnterEnter<q_next/gradients/q_next/rnn/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
П
8q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1SelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2:q_next/gradients/q_next/rnn/while/Select_1_grad/zeros_likeIq_next/gradients/q_next/rnn/while/Merge_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
М
@q_next/gradients/q_next/rnn/while/Select_1_grad/tuple/group_depsNoOp7^q_next/gradients/q_next/rnn/while/Select_1_grad/Select9^q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1
Э
Hq_next/gradients/q_next/rnn/while/Select_1_grad/tuple/control_dependencyIdentity6q_next/gradients/q_next/rnn/while/Select_1_grad/SelectA^q_next/gradients/q_next/rnn/while/Select_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_1_grad/Select*(
_output_shapes
:џџџџџџџџџ
г
Jq_next/gradients/q_next/rnn/while/Select_1_grad/tuple/control_dependency_1Identity8q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1A^q_next/gradients/q_next/rnn/while/Select_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1*(
_output_shapes
:џџџџџџџџџ
С
:q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like	ZerosLikeEq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
Л
@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@q_next/rnn/while/Identity_4*
valueB :
џџџџџџџџџ
ќ
@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/f_accStackV2@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/Const*
	elem_type0*.
_class$
" loc:@q_next/rnn/while/Identity_4*

stack_name *
_output_shapes
:

@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/EnterEnter@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context

Fq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/Enterq_next/rnn/while/Identity_4^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0*(
_output_shapes
:џџџџџџџџџ
Њ
Kq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant(
Н
6q_next/gradients/q_next/rnn/while/Select_2_grad/SelectSelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2Iq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependency_1:q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
П
8q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1SelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2:q_next/gradients/q_next/rnn/while/Select_2_grad/zeros_likeIq_next/gradients/q_next/rnn/while/Merge_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
М
@q_next/gradients/q_next/rnn/while/Select_2_grad/tuple/group_depsNoOp7^q_next/gradients/q_next/rnn/while/Select_2_grad/Select9^q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1
Э
Hq_next/gradients/q_next/rnn/while/Select_2_grad/tuple/control_dependencyIdentity6q_next/gradients/q_next/rnn/while/Select_2_grad/SelectA^q_next/gradients/q_next/rnn/while/Select_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_2_grad/Select*(
_output_shapes
:џџџџџџџџџ
г
Jq_next/gradients/q_next/rnn/while/Select_2_grad/tuple/control_dependency_1Identity8q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1A^q_next/gradients/q_next/rnn/while/Select_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1*(
_output_shapes
:џџџџџџџџџ
Я
8q_next/gradients/q_next/rnn/while/Select_grad/zeros_like	ZerosLike>q_next/gradients/q_next/rnn/while/Select_grad/zeros_like/Enter^q_next/gradients/Sub*
T0*(
_output_shapes
:џџџџџџџџџ
ћ
>q_next/gradients/q_next/rnn/while/Select_grad/zeros_like/EnterEnterq_next/rnn/zeros*
T0*
is_constant(*
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
г
4q_next/gradients/q_next/rnn/while/Select_grad/SelectSelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV2cq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency8q_next/gradients/q_next/rnn/while/Select_grad/zeros_like*
T0*(
_output_shapes
:џџџџџџџџџ
е
6q_next/gradients/q_next/rnn/while/Select_grad/Select_1SelectAq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPopV28q_next/gradients/q_next/rnn/while/Select_grad/zeros_likecq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
Ж
>q_next/gradients/q_next/rnn/while/Select_grad/tuple/group_depsNoOp5^q_next/gradients/q_next/rnn/while/Select_grad/Select7^q_next/gradients/q_next/rnn/while/Select_grad/Select_1
Х
Fq_next/gradients/q_next/rnn/while/Select_grad/tuple/control_dependencyIdentity4q_next/gradients/q_next/rnn/while/Select_grad/Select?^q_next/gradients/q_next/rnn/while/Select_grad/tuple/group_deps*
T0*G
_class=
;9loc:@q_next/gradients/q_next/rnn/while/Select_grad/Select*(
_output_shapes
:џџџџџџџџџ
Ы
Hq_next/gradients/q_next/rnn/while/Select_grad/tuple/control_dependency_1Identity6q_next/gradients/q_next/rnn/while/Select_grad/Select_1?^q_next/gradients/q_next/rnn/while/Select_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_grad/Select_1

9q_next/gradients/q_next/rnn/while/Select/Enter_grad/ShapeShapeq_next/rnn/zeros*
_output_shapes
:*
T0*
out_type0

?q_next/gradients/q_next/rnn/while/Select/Enter_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

9q_next/gradients/q_next/rnn/while/Select/Enter_grad/zerosFill9q_next/gradients/q_next/rnn/while/Select/Enter_grad/Shape?q_next/gradients/q_next/rnn/while/Select/Enter_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0

9q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_accEnter9q_next/gradients/q_next/rnn/while/Select/Enter_grad/zeros*
parallel_iterations *(
_output_shapes
:џџџџџџџџџ*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant( 

;q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_acc_1Merge9q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_accAq_next/gradients/q_next/rnn/while/Select/Enter_grad/NextIteration*
N**
_output_shapes
:џџџџџџџџџ: *
T0
ф
:q_next/gradients/q_next/rnn/while/Select/Enter_grad/SwitchSwitch;q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_acc_1q_next/gradients/b_count_2*
T0*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
ї
7q_next/gradients/q_next/rnn/while/Select/Enter_grad/AddAdd<q_next/gradients/q_next/rnn/while/Select/Enter_grad/Switch:1Fq_next/gradients/q_next/rnn/while/Select_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
О
Aq_next/gradients/q_next/rnn/while/Select/Enter_grad/NextIterationNextIteration7q_next/gradients/q_next/rnn/while/Select/Enter_grad/Add*(
_output_shapes
:џџџџџџџџџ*
T0
В
;q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_acc_2Exit:q_next/gradients/q_next/rnn/while/Select/Enter_grad/Switch*
T0*(
_output_shapes
:џџџџџџџџџ
М
q_next/gradients/AddNAddNJq_next/gradients/q_next/rnn/while/Select_2_grad/tuple/control_dependency_1Hq_next/gradients/q_next/rnn/while/Select_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Select_2_grad/Select_1*
N*(
_output_shapes
:џџџџџџџџџ
 
<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/ShapeShape$q_next/rnn/while/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape_1Shape!q_next/rnn/while/lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
ж
Lq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsWq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ю
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
В
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ш
Xq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape^q_next/gradients/Add*
_output_shapes
:*
swap_memory( *
T0

Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
]q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ђ
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
Ж
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1EnterTq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ю
Zq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1>q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape_1^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
в
_q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
в
:q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/MulMulq_next/gradients/AddNEq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
С
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@q_next/rnn/while/lstm_cell/Tanh_1*
valueB :
џџџџџџџџџ

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/f_accStackV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/Const*4
_class*
(&loc:@q_next/rnn/while/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context

Fq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/Enter!q_next/rnn/while/lstm_cell/Tanh_1^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0*(
_output_shapes
:џџџџџџџџџ
Њ
Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/SumSum:q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/MulLq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/ReshapeReshape:q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/SumWq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ж
<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1MulGq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2q_next/gradients/AddN*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *7
_class-
+)loc:@q_next/rnn/while/lstm_cell/Sigmoid_2*
valueB :
џџџџџџџџџ

Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/Const*7
_class-
+)loc:@q_next/rnn/while/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0

Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterBq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(

Hq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter$q_next/rnn/while/lstm_cell/Sigmoid_2^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
і
Gq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0*(
_output_shapes
:џџџџџџџџџ
Ў
Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterBq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Sum_1Sum<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1Nq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape_1Reshape<q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Sum_1Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Gq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/ReshapeA^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape_1
ы
Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentity>q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/ReshapeH^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1Identity@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape_1H^q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
и
?q_next/gradients/q_next/rnn/while/Switch_2_grad_1/NextIterationNextIterationeq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
_output_shapes
: *
T0
}
,q_next/gradients/q_next/rnn/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
к
*q_next/gradients/q_next/rnn/zeros_grad/SumSum;q_next/gradients/q_next/rnn/while/Select/Enter_grad/b_acc_2,q_next/gradients/q_next/rnn/zeros_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Ђ
Fq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradGq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

@q_next/gradients/q_next/rnn/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradEq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ж
q_next/gradients/AddN_1AddNJq_next/gradients/q_next/rnn/while/Select_1_grad/tuple/control_dependency_1@q_next/gradients/q_next/rnn/while/lstm_cell/Tanh_1_grad/TanhGrad*
N*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@q_next/gradients/q_next/rnn/while/Select_1_grad/Select_1

<q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/ShapeShapeq_next/rnn/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:

>q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape_1Shape q_next/rnn/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
ж
Lq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsWq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2Yq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ю
Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
В
Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ш
Xq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter<q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
]q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant(
ђ
Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape_1*
valueB :
џџџџџџџџџ
Ч
Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:
Ж
Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1EnterTq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ю
Zq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1>q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape_1^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
в
_q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ш
:q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/SumSumq_next/gradients/AddN_1Lq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/ReshapeReshape:q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/SumWq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ь
<q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Sum_1Sumq_next/gradients/AddN_1Nq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ѕ
@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape_1Reshape<q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Sum_1Yq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Gq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/ReshapeA^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape_1
ы
Oq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentity>q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/ReshapeH^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1Identity@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape_1H^q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Reshape_1

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/ShapeShape"q_next/rnn/while/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape_1Shapeq_next/rnn/while/Identity_3*
_output_shapes
:*
T0*
out_type0
а
Jq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsUq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ъ
Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *M
_classC
A?loc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape*
valueB :
џџџџџџџџџ
Л
Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*M
_classC
A?loc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
Ў
Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterPq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Т
Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Pq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape^q_next/gradients/Add*
_output_shapes
:*
swap_memory( *
T0

Uq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2[q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
Ъ
[q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterPq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ю
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
С
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
В
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1EnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ш
Xq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1<q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape_1^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2]q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_next/gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
]q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

8q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/MulMulOq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyEq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0

8q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/SumSum8q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/MulJq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/ReshapeReshape8q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/SumUq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1MulEq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2Oq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
Т
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/ConstConst*5
_class+
)'loc:@q_next/rnn/while/lstm_cell/Sigmoid*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/f_accStackV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/Const*
	elem_type0*5
_class+
)'loc:@q_next/rnn/while/lstm_cell/Sigmoid*

stack_name *
_output_shapes
:

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context

Fq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/Enter"q_next/rnn/while/lstm_cell/Sigmoid^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
ђ
Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Sum_1Sum:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1Lq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape_1Reshape:q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Sum_1Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Э
Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/group_depsNoOp=^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape?^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape_1
у
Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/control_dependencyIdentity<q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/ReshapeF^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape
щ
Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1Identity>q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape_1F^q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
 
<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/ShapeShape$q_next/rnn/while/lstm_cell/Sigmoid_1*
_output_shapes
:*
T0*
out_type0

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape_1Shapeq_next/rnn/while/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
ж
Lq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsWq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ю
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*
dtype0*
_output_shapes
: *O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape*
valueB :
џџџџџџџџџ
С
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape*

stack_name *
_output_shapes
:
В
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context*
T0*
is_constant(
Ш
Xq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape^q_next/gradients/Add*
_output_shapes
:*
swap_memory( *
T0

Wq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2]q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0*
_output_shapes
:
Ю
]q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterRq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
ђ
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ч
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*

stack_name *
_output_shapes
:*
	elem_type0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape_1
Ж
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1EnterTq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ю
Zq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1>q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape_1^q_next/gradients/Add*
T0*
_output_shapes
:*
swap_memory( 

Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2_q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^q_next/gradients/Sub*
	elem_type0*
_output_shapes
:
в
_q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterTq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/MulMulQq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:џџџџџџџџџ
П
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/ConstConst*2
_class(
&$loc:@q_next/rnn/while/lstm_cell/Tanh*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/f_accStackV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/Const*2
_class(
&$loc:@q_next/rnn/while/lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0

@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context

Fq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/Enterq_next/rnn/while/lstm_cell/Tanh^q_next/gradients/Add*(
_output_shapes
:џџџџџџџџџ*
swap_memory( *
T0
ђ
Eq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Њ
Kq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnter@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant(

:q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/SumSum:q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/MulLq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/ReshapeReshape:q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/SumWq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1MulGq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*7
_class-
+)loc:@q_next/rnn/while/lstm_cell/Sigmoid_1*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*7
_class-
+)loc:@q_next/rnn/while/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:

Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterBq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context

Hq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter$q_next/rnn/while/lstm_cell/Sigmoid_1^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ*
swap_memory( 
і
Gq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ*
	elem_type0
Ў
Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterBq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Sum_1Sum<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1Nq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape_1Reshape<q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Sum_1Yq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Gq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/ReshapeA^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape_1
ы
Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentity>q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/ReshapeH^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape
ё
Qq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1Identity@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape_1H^q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

Dq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradEq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2Mq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
С
q_next/gradients/AddN_2AddNHq_next/gradients/q_next/rnn/while/Select_1_grad/tuple/control_dependencyOq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_1_grad/Select*
N*(
_output_shapes
:џџџџџџџџџ
Ђ
Fq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradGq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Oq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

>q_next/gradients/q_next/rnn/while/lstm_cell/Tanh_grad/TanhGradTanhGradEq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:џџџџџџџџџ

:q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/ShapeShape"q_next/rnn/while/lstm_cell/split:2*
_output_shapes
:*
T0*
out_type0

<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape_1Const^q_next/gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
Е
Jq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsUq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*M
_classC
A?loc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*M
_classC
A?loc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
Ў
Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterPq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Т
Vq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Pq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape^q_next/gradients/Add*
_output_shapes
:*
swap_memory( *
T0

Uq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2[q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^q_next/gradients/Sub*
_output_shapes
:*
	elem_type0
Ъ
[q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterPq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

8q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/SumSumDq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradJq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/ReshapeReshape8q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/SumUq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

:q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Sum_1SumDq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradLq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ђ
>q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape_1Reshape:q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Sum_1<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Э
Eq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/group_depsNoOp=^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape?^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape_1
у
Mq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/control_dependencyIdentity<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/ReshapeF^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
з
Oq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/control_dependency_1Identity>q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape_1F^q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 

?q_next/gradients/q_next/rnn/while/Switch_3_grad_1/NextIterationNextIterationq_next/gradients/AddN_2*
T0*(
_output_shapes
:џџџџџџџџџ
ѕ
=q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concatConcatV2Fq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGrad>q_next/gradients/q_next/rnn/while/lstm_cell/Tanh_grad/TanhGradMq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/tuple/control_dependencyFq_next/gradients/q_next/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradCq_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat/Const*
N*(
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0

Cq_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat/ConstConst^q_next/gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
Я
Dq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad=q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
и
Iq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpE^q_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad>^q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat
э
Qq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity=q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concatJ^q_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_next/gradients/q_next/rnn/while/lstm_cell/split_grad/concat*(
_output_shapes
:џџџџџџџџџ
№
Sq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityDq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradJ^q_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@q_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
К
>q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMulMatMulQq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyDq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul/Enter*
T0*(
_output_shapes
:џџџџџџџџџ *
transpose_a( *
transpose_b(

Dq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul/EnterEnter q_next/rnn/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
 *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
Л
@q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1MatMulKq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
 *
transpose_a(*
transpose_b( *
T0
Ч
Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*4
_class*
(&loc:@q_next/rnn/while/lstm_cell/concat*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*4
_class*
(&loc:@q_next/rnn/while/lstm_cell/concat

Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterFq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
Ѓ
Lq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Fq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter!q_next/rnn/while/lstm_cell/concat^q_next/gradients/Add*(
_output_shapes
:џџџџџџџџџ *
swap_memory( *
T0
ў
Kq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Qq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^q_next/gradients/Sub*(
_output_shapes
:џџџџџџџџџ *
	elem_type0
Ж
Qq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterFq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
д
Hq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMulA^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1
э
Pq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentity>q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMulI^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ 
ы
Rq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1Identity@q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1I^q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
 *
T0*S
_classI
GEloc:@q_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1

Dq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes	
:
Њ
Fq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterDq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

Fq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeFq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Lq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
р
Eq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchFq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2q_next/gradients/b_count_2*"
_output_shapes
::*
T0

Bq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/AddAddGq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch:1Sq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ч
Lq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationBq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
Л
Fq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitEq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:

=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ConstConst^q_next/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :

<q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/RankConst^q_next/gradients/Sub*
dtype0*
_output_shapes
: *
value	B :
х
;q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/modFloorMod=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Const<q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0

=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeShape"q_next/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:

>q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeNShapeNIq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2Eq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPopV2*
T0*
out_type0*
N* 
_output_shapes
::
Ц
Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/ConstConst*
dtype0*
_output_shapes
: *5
_class+
)'loc:@q_next/rnn/while/TensorArrayReadV3*
valueB :
џџџџџџџџџ

Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/Const*
	elem_type0*5
_class+
)'loc:@q_next/rnn/while/TensorArrayReadV3*

stack_name *
_output_shapes
:

Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/EnterEnterDq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name q_next/rnn/while/while_context
 
Jq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/Enter"q_next/rnn/while/TensorArrayReadV3^q_next/gradients/Add*
T0*(
_output_shapes
:џџџџџџџџџ *
swap_memory( 
њ
Iq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Oq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^q_next/gradients/Sub*
	elem_type0*(
_output_shapes
:џџџџџџџџџ 
В
Oq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterDq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context
О
Dq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ConcatOffsetConcatOffset;q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/mod>q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN@q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
ц
=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/SliceSlicePq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyDq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ConcatOffset>q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ь
?q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice_1SlicePq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyFq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ConcatOffset:1@q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
в
Hq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/group_depsNoOp>^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice@^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice_1
ы
Pq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyIdentity=q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/SliceI^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice*(
_output_shapes
:џџџџџџџџџ 
ё
Rq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1Identity?q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice_1I^q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*R
_classH
FDloc:@q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Slice_1*(
_output_shapes
:џџџџџџџџџ

Cq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
 *    *
dtype0* 
_output_shapes
:
 
­
Eq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterCq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
 *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

Eq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeEq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1Kq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
 : *
T0
ш
Dq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchEq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2q_next/gradients/b_count_2*
T0*,
_output_shapes
:
 :
 

Aq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/AddAddFq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/Switch:1Rq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
 *
T0
Ъ
Kq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationAq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
 
О
Eq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitDq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/Switch* 
_output_shapes
:
 *
T0
Х
Zq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3`q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterbq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^q_next/gradients/Sub*
_output_shapes

:: *;
_class1
/-loc:@q_next/rnn/while/TensorArrayReadV3/Enter*
sourceq_next/gradients
д
`q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterq_next/rnn/TensorArray_1*
is_constant(*
_output_shapes
:*?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*;
_class1
/-loc:@q_next/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations 
џ
bq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterEq_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*;
_class1
/-loc:@q_next/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context

Vq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitybq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1[^q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*;
_class1
/-loc:@q_next/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 

\q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Zq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3gq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Pq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyVq_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
Ф
q_next/gradients/AddN_3AddNHq_next/gradients/q_next/rnn/while/Select_2_grad/tuple/control_dependencyRq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*I
_class?
=;loc:@q_next/gradients/q_next/rnn/while/Select_2_grad/Select

Fq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
Hq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterFq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
parallel_iterations *
_output_shapes
: *?

frame_name1/q_next/gradients/q_next/rnn/while/while_context*
T0*
is_constant( 

Hq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeHq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Nq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
к
Gq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchHq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2q_next/gradients/b_count_2*
T0*
_output_shapes
: : 

Dq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddIq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1\q_next/gradients/q_next/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ц
Nq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationDq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
К
Hq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitGq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 

?q_next/gradients/q_next/rnn/while/Switch_4_grad_1/NextIterationNextIterationq_next/gradients/AddN_3*
T0*(
_output_shapes
:џџџџџџџџџ
п
}q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3q_next/rnn/TensorArray_1Hq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*+
_class!
loc:@q_next/rnn/TensorArray_1*
sourceq_next/gradients*
_output_shapes

:: 

yq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityHq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3~^q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@q_next/rnn/TensorArray_1*
_output_shapes
: 

oq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3}q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3#q_next/rnn/TensorArrayUnstack/rangeyq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
element_shape:
Б
lq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpp^q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3I^q_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
Ѕ
tq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityoq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3m^q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*
_classx
vtloc:@q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ 
Й
vq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityHq_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3m^q_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@q_next/gradients/q_next/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 

<q_next/gradients/q_next/rnn/transpose_grad/InvertPermutationInvertPermutationq_next/rnn/concat*
T0*
_output_shapes
:
Т
4q_next/gradients/q_next/rnn/transpose_grad/transpose	Transposetq_next/gradients/q_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency<q_next/gradients/q_next/rnn/transpose_grad/InvertPermutation*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ *
Tperm0*
T0
z
,q_next/gradients/q_next/Reshape_1_grad/ShapeShapeq_next/Reshape*
T0*
out_type0*
_output_shapes
:
о
.q_next/gradients/q_next/Reshape_1_grad/ReshapeReshape4q_next/gradients/q_next/rnn/transpose_grad/transpose,q_next/gradients/q_next/Reshape_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ 
w
*q_next/gradients/q_next/Reshape_grad/ShapeShapeq_next/Relu_1*
T0*
out_type0*
_output_shapes
:
л
,q_next/gradients/q_next/Reshape_grad/ReshapeReshape.q_next/gradients/q_next/Reshape_1_grad/Reshape*q_next/gradients/q_next/Reshape_grad/Shape*/
_output_shapes
:џџџџџџџџџ		 *
T0*
Tshape0
Џ
,q_next/gradients/q_next/Relu_1_grad/ReluGradReluGrad,q_next/gradients/q_next/Reshape_grad/Reshapeq_next/Relu_1*
T0*/
_output_shapes
:џџџџџџџџџ		 
Џ
6q_next/gradients/q_next/conv2/BiasAdd_grad/BiasAddGradBiasAddGrad,q_next/gradients/q_next/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
Ћ
;q_next/gradients/q_next/conv2/BiasAdd_grad/tuple/group_depsNoOp-^q_next/gradients/q_next/Relu_1_grad/ReluGrad7^q_next/gradients/q_next/conv2/BiasAdd_grad/BiasAddGrad
Ж
Cq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependencyIdentity,q_next/gradients/q_next/Relu_1_grad/ReluGrad<^q_next/gradients/q_next/conv2/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@q_next/gradients/q_next/Relu_1_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ		 
З
Eq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependency_1Identity6q_next/gradients/q_next/conv2/BiasAdd_grad/BiasAddGrad<^q_next/gradients/q_next/conv2/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *
T0*I
_class?
=;loc:@q_next/gradients/q_next/conv2/BiasAdd_grad/BiasAddGrad
Ѕ
0q_next/gradients/q_next/conv2/Conv2D_grad/ShapeNShapeNq_next/Reluq_next/conv2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::

/q_next/gradients/q_next/conv2/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
Љ
=q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput0q_next/gradients/q_next/conv2/Conv2D_grad/ShapeNq_next/conv2/kernel/readCq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
љ
>q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterq_next/Relu/q_next/gradients/q_next/conv2/Conv2D_grad/ConstCq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
: *
	dilations
*
T0
У
:q_next/gradients/q_next/conv2/Conv2D_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropFilter>^q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropInput
ж
Bq_next/gradients/q_next/conv2/Conv2D_grad/tuple/control_dependencyIdentity=q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropInput;^q_next/gradients/q_next/conv2/Conv2D_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ
б
Dq_next/gradients/q_next/conv2/Conv2D_grad/tuple/control_dependency_1Identity>q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropFilter;^q_next/gradients/q_next/conv2/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
С
*q_next/gradients/q_next/Relu_grad/ReluGradReluGradBq_next/gradients/q_next/conv2/Conv2D_grad/tuple/control_dependencyq_next/Relu*
T0*/
_output_shapes
:џџџџџџџџџ
­
6q_next/gradients/q_next/conv1/BiasAdd_grad/BiasAddGradBiasAddGrad*q_next/gradients/q_next/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
Љ
;q_next/gradients/q_next/conv1/BiasAdd_grad/tuple/group_depsNoOp+^q_next/gradients/q_next/Relu_grad/ReluGrad7^q_next/gradients/q_next/conv1/BiasAdd_grad/BiasAddGrad
В
Cq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependencyIdentity*q_next/gradients/q_next/Relu_grad/ReluGrad<^q_next/gradients/q_next/conv1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@q_next/gradients/q_next/Relu_grad/ReluGrad*/
_output_shapes
:џџџџџџџџџ
З
Eq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependency_1Identity6q_next/gradients/q_next/conv1/BiasAdd_grad/BiasAddGrad<^q_next/gradients/q_next/conv1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@q_next/gradients/q_next/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ї
0q_next/gradients/q_next/conv1/Conv2D_grad/ShapeNShapeNq_next/statesq_next/conv1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::

/q_next/gradients/q_next/conv1/Conv2D_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
Љ
=q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput0q_next/gradients/q_next/conv1/Conv2D_grad/ShapeNq_next/conv1/kernel/readCq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependency*
paddingVALID*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
ћ
>q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterq_next/states/q_next/gradients/q_next/conv1/Conv2D_grad/ConstCq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations

У
:q_next/gradients/q_next/conv1/Conv2D_grad/tuple/group_depsNoOp?^q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropFilter>^q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropInput
ж
Bq_next/gradients/q_next/conv1/Conv2D_grad/tuple/control_dependencyIdentity=q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropInput;^q_next/gradients/q_next/conv1/Conv2D_grad/tuple/group_deps*
T0*P
_classF
DBloc:@q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџTT
б
Dq_next/gradients/q_next/conv1/Conv2D_grad/tuple/control_dependency_1Identity>q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropFilter;^q_next/gradients/q_next/conv1/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@q_next/gradients/q_next/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

 q_next/beta1_power/initial_valueConst* 
_class
loc:@q_next/biases*
valueB
 *fff?*
dtype0*
_output_shapes
: 

q_next/beta1_power
VariableV2*
shared_name * 
_class
loc:@q_next/biases*
	container *
shape: *
dtype0*
_output_shapes
: 
Х
q_next/beta1_power/AssignAssignq_next/beta1_power q_next/beta1_power/initial_value*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: *
use_locking(
z
q_next/beta1_power/readIdentityq_next/beta1_power*
T0* 
_class
loc:@q_next/biases*
_output_shapes
: 

 q_next/beta2_power/initial_valueConst* 
_class
loc:@q_next/biases*
valueB
 *wО?*
dtype0*
_output_shapes
: 

q_next/beta2_power
VariableV2* 
_class
loc:@q_next/biases*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Х
q_next/beta2_power/AssignAssignq_next/beta2_power q_next/beta2_power/initial_value*
use_locking(*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: 
z
q_next/beta2_power/readIdentityq_next/beta2_power*
_output_shapes
: *
T0* 
_class
loc:@q_next/biases
Т
Aq_next/q_next/conv1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_next/conv1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Є
7q_next/q_next/conv1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_next/conv1/kernel*
valueB
 *    
 
1q_next/q_next/conv1/kernel/Adam/Initializer/zerosFillAq_next/q_next/conv1/kernel/Adam/Initializer/zeros/shape_as_tensor7q_next/q_next/conv1/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@q_next/conv1/kernel*

index_type0*&
_output_shapes
:
Ы
q_next/q_next/conv1/kernel/Adam
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *&
_class
loc:@q_next/conv1/kernel*
	container *
shape:

&q_next/q_next/conv1/kernel/Adam/AssignAssignq_next/q_next/conv1/kernel/Adam1q_next/q_next/conv1/kernel/Adam/Initializer/zeros*
T0*&
_class
loc:@q_next/conv1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
Њ
$q_next/q_next/conv1/kernel/Adam/readIdentityq_next/q_next/conv1/kernel/Adam*
T0*&
_class
loc:@q_next/conv1/kernel*&
_output_shapes
:
Ф
Cq_next/q_next/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*&
_class
loc:@q_next/conv1/kernel*%
valueB"            
І
9q_next/q_next/conv1/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@q_next/conv1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
І
3q_next/q_next/conv1/kernel/Adam_1/Initializer/zerosFillCq_next/q_next/conv1/kernel/Adam_1/Initializer/zeros/shape_as_tensor9q_next/q_next/conv1/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@q_next/conv1/kernel*

index_type0*&
_output_shapes
:
Э
!q_next/q_next/conv1/kernel/Adam_1
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *&
_class
loc:@q_next/conv1/kernel*
	container *
shape:

(q_next/q_next/conv1/kernel/Adam_1/AssignAssign!q_next/q_next/conv1/kernel/Adam_13q_next/q_next/conv1/kernel/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_next/conv1/kernel
Ў
&q_next/q_next/conv1/kernel/Adam_1/readIdentity!q_next/q_next/conv1/kernel/Adam_1*&
_output_shapes
:*
T0*&
_class
loc:@q_next/conv1/kernel
Ђ
/q_next/q_next/conv1/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*$
_class
loc:@q_next/conv1/bias*
valueB*    
Џ
q_next/q_next/conv1/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@q_next/conv1/bias
ђ
$q_next/q_next/conv1/bias/Adam/AssignAssignq_next/q_next/conv1/bias/Adam/q_next/q_next/conv1/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@q_next/conv1/bias*
validate_shape(*
_output_shapes
:

"q_next/q_next/conv1/bias/Adam/readIdentityq_next/q_next/conv1/bias/Adam*
T0*$
_class
loc:@q_next/conv1/bias*
_output_shapes
:
Є
1q_next/q_next/conv1/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@q_next/conv1/bias*
valueB*    *
dtype0*
_output_shapes
:
Б
q_next/q_next/conv1/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@q_next/conv1/bias*
	container 
ј
&q_next/q_next/conv1/bias/Adam_1/AssignAssignq_next/q_next/conv1/bias/Adam_11q_next/q_next/conv1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_next/conv1/bias

$q_next/q_next/conv1/bias/Adam_1/readIdentityq_next/q_next/conv1/bias/Adam_1*
T0*$
_class
loc:@q_next/conv1/bias*
_output_shapes
:
Т
Aq_next/q_next/conv2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_next/conv2/kernel*%
valueB"             *
dtype0*
_output_shapes
:
Є
7q_next/q_next/conv2/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@q_next/conv2/kernel*
valueB
 *    
 
1q_next/q_next/conv2/kernel/Adam/Initializer/zerosFillAq_next/q_next/conv2/kernel/Adam/Initializer/zeros/shape_as_tensor7q_next/q_next/conv2/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@q_next/conv2/kernel*

index_type0*&
_output_shapes
: 
Ы
q_next/q_next/conv2/kernel/Adam
VariableV2*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_next/conv2/kernel

&q_next/q_next/conv2/kernel/Adam/AssignAssignq_next/q_next/conv2/kernel/Adam1q_next/q_next/conv2/kernel/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel
Њ
$q_next/q_next/conv2/kernel/Adam/readIdentityq_next/q_next/conv2/kernel/Adam*
T0*&
_class
loc:@q_next/conv2/kernel*&
_output_shapes
: 
Ф
Cq_next/q_next/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@q_next/conv2/kernel*%
valueB"             *
dtype0*
_output_shapes
:
І
9q_next/q_next/conv2/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@q_next/conv2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
І
3q_next/q_next/conv2/kernel/Adam_1/Initializer/zerosFillCq_next/q_next/conv2/kernel/Adam_1/Initializer/zeros/shape_as_tensor9q_next/q_next/conv2/kernel/Adam_1/Initializer/zeros/Const*&
_output_shapes
: *
T0*&
_class
loc:@q_next/conv2/kernel*

index_type0
Э
!q_next/q_next/conv2/kernel/Adam_1
VariableV2*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name *&
_class
loc:@q_next/conv2/kernel

(q_next/q_next/conv2/kernel/Adam_1/AssignAssign!q_next/q_next/conv2/kernel/Adam_13q_next/q_next/conv2/kernel/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel
Ў
&q_next/q_next/conv2/kernel/Adam_1/readIdentity!q_next/q_next/conv2/kernel/Adam_1*
T0*&
_class
loc:@q_next/conv2/kernel*&
_output_shapes
: 
Ђ
/q_next/q_next/conv2/bias/Adam/Initializer/zerosConst*$
_class
loc:@q_next/conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Џ
q_next/q_next/conv2/bias/Adam
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@q_next/conv2/bias*
	container 
ђ
$q_next/q_next/conv2/bias/Adam/AssignAssignq_next/q_next/conv2/bias/Adam/q_next/q_next/conv2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias

"q_next/q_next/conv2/bias/Adam/readIdentityq_next/q_next/conv2/bias/Adam*
T0*$
_class
loc:@q_next/conv2/bias*
_output_shapes
: 
Є
1q_next/q_next/conv2/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@q_next/conv2/bias*
valueB *    *
dtype0*
_output_shapes
: 
Б
q_next/q_next/conv2/bias/Adam_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *$
_class
loc:@q_next/conv2/bias
ј
&q_next/q_next/conv2/bias/Adam_1/AssignAssignq_next/q_next/conv2/bias/Adam_11q_next/q_next/conv2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias

$q_next/q_next/conv2/bias/Adam_1/readIdentityq_next/q_next/conv2/bias/Adam_1*
T0*$
_class
loc:@q_next/conv2/bias*
_output_shapes
: 
Ъ
Iq_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Д
?q_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
К
9q_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zerosFillIq_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensor?q_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*

index_type0* 
_output_shapes
:
 
Я
'q_next/q_next/rnn/lstm_cell/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
 *
shared_name *.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
	container *
shape:
 
 
.q_next/q_next/rnn/lstm_cell/kernel/Adam/AssignAssign'q_next/q_next/rnn/lstm_cell/kernel/Adam9q_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
 *
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel
М
,q_next/q_next/rnn/lstm_cell/kernel/Adam/readIdentity'q_next/q_next/rnn/lstm_cell/kernel/Adam*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel* 
_output_shapes
:
 
Ь
Kq_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ж
Aq_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
;q_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zerosFillKq_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorAq_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
 *
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*

index_type0
б
)q_next/q_next/rnn/lstm_cell/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
 *
shared_name *.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
	container *
shape:
 
І
0q_next/q_next/rnn/lstm_cell/kernel/Adam_1/AssignAssign)q_next/q_next/rnn/lstm_cell/kernel/Adam_1;q_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
Р
.q_next/q_next/rnn/lstm_cell/kernel/Adam_1/readIdentity)q_next/q_next/rnn/lstm_cell/kernel/Adam_1*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel* 
_output_shapes
:
 
Р
Gq_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB:*
dtype0*
_output_shapes
:
А
=q_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB
 *    
­
7q_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zerosFillGq_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensor=q_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
С
%q_next/q_next/rnn/lstm_cell/bias/Adam
VariableV2*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

,q_next/q_next/rnn/lstm_cell/bias/Adam/AssignAssign%q_next/q_next/rnn/lstm_cell/bias/Adam7q_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Б
*q_next/q_next/rnn/lstm_cell/bias/Adam/readIdentity%q_next/q_next/rnn/lstm_cell/bias/Adam*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
_output_shapes	
:
Т
Iq_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB:
В
?q_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
valueB
 *    
Г
9q_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zerosFillIq_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensor?q_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*

index_type0*
_output_shapes	
:
У
'q_next/q_next/rnn/lstm_cell/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
	container *
shape:

.q_next/q_next/rnn/lstm_cell/bias/Adam_1/AssignAssign'q_next/q_next/rnn/lstm_cell/bias/Adam_19q_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias
Е
,q_next/q_next/rnn/lstm_cell/bias/Adam_1/readIdentity'q_next/q_next/rnn/lstm_cell/bias/Adam_1*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
_output_shapes	
:
А
<q_next/q_next/weights/Adam/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@q_next/weights*
valueB"      *
dtype0*
_output_shapes
:

2q_next/q_next/weights/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *!
_class
loc:@q_next/weights*
valueB
 *    

,q_next/q_next/weights/Adam/Initializer/zerosFill<q_next/q_next/weights/Adam/Initializer/zeros/shape_as_tensor2q_next/q_next/weights/Adam/Initializer/zeros/Const*
T0*!
_class
loc:@q_next/weights*

index_type0*
_output_shapes
:	
Г
q_next/q_next/weights/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *!
_class
loc:@q_next/weights*
	container *
shape:	
ы
!q_next/q_next/weights/Adam/AssignAssignq_next/q_next/weights/Adam,q_next/q_next/weights/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	

q_next/q_next/weights/Adam/readIdentityq_next/q_next/weights/Adam*
T0*!
_class
loc:@q_next/weights*
_output_shapes
:	
В
>q_next/q_next/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@q_next/weights*
valueB"      *
dtype0*
_output_shapes
:

4q_next/q_next/weights/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@q_next/weights*
valueB
 *    *
dtype0*
_output_shapes
: 

.q_next/q_next/weights/Adam_1/Initializer/zerosFill>q_next/q_next/weights/Adam_1/Initializer/zeros/shape_as_tensor4q_next/q_next/weights/Adam_1/Initializer/zeros/Const*
_output_shapes
:	*
T0*!
_class
loc:@q_next/weights*

index_type0
Е
q_next/q_next/weights/Adam_1
VariableV2*!
_class
loc:@q_next/weights*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
ё
#q_next/q_next/weights/Adam_1/AssignAssignq_next/q_next/weights/Adam_1.q_next/q_next/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	

!q_next/q_next/weights/Adam_1/readIdentityq_next/q_next/weights/Adam_1*
T0*!
_class
loc:@q_next/weights*
_output_shapes
:	

+q_next/q_next/biases/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:* 
_class
loc:@q_next/biases*
valueB*    
Ї
q_next/q_next/biases/Adam
VariableV2*
shared_name * 
_class
loc:@q_next/biases*
	container *
shape:*
dtype0*
_output_shapes
:
т
 q_next/q_next/biases/Adam/AssignAssignq_next/q_next/biases/Adam+q_next/q_next/biases/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
:

q_next/q_next/biases/Adam/readIdentityq_next/q_next/biases/Adam*
T0* 
_class
loc:@q_next/biases*
_output_shapes
:

-q_next/q_next/biases/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:* 
_class
loc:@q_next/biases*
valueB*    
Љ
q_next/q_next/biases/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@q_next/biases*
	container 
ш
"q_next/q_next/biases/Adam_1/AssignAssignq_next/q_next/biases/Adam_1-q_next/q_next/biases/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_next/biases

 q_next/q_next/biases/Adam_1/readIdentityq_next/q_next/biases/Adam_1*
T0* 
_class
loc:@q_next/biases*
_output_shapes
:
^
q_next/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o9
V
q_next/Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
V
q_next/Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
X
q_next/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
ф
0q_next/Adam/update_q_next/conv1/kernel/ApplyAdam	ApplyAdamq_next/conv1/kernelq_next/q_next/conv1/kernel/Adam!q_next/q_next/conv1/kernel/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonDq_next/gradients/q_next/conv1/Conv2D_grad/tuple/control_dependency_1*
T0*&
_class
loc:@q_next/conv1/kernel*
use_nesterov( *&
_output_shapes
:*
use_locking( 
Я
.q_next/Adam/update_q_next/conv1/bias/ApplyAdam	ApplyAdamq_next/conv1/biasq_next/q_next/conv1/bias/Adamq_next/q_next/conv1/bias/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonEq_next/gradients/q_next/conv1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*$
_class
loc:@q_next/conv1/bias
ф
0q_next/Adam/update_q_next/conv2/kernel/ApplyAdam	ApplyAdamq_next/conv2/kernelq_next/q_next/conv2/kernel/Adam!q_next/q_next/conv2/kernel/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonDq_next/gradients/q_next/conv2/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
: *
use_locking( *
T0*&
_class
loc:@q_next/conv2/kernel
Я
.q_next/Adam/update_q_next/conv2/bias/ApplyAdam	ApplyAdamq_next/conv2/biasq_next/q_next/conv2/bias/Adamq_next/q_next/conv2/bias/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonEq_next/gradients/q_next/conv2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@q_next/conv2/bias*
use_nesterov( *
_output_shapes
: 

8q_next/Adam/update_q_next/rnn/lstm_cell/kernel/ApplyAdam	ApplyAdamq_next/rnn/lstm_cell/kernel'q_next/q_next/rnn/lstm_cell/kernel/Adam)q_next/q_next/rnn/lstm_cell/kernel/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonEq_next/gradients/q_next/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
use_nesterov( * 
_output_shapes
:
 *
use_locking( 
љ
6q_next/Adam/update_q_next/rnn/lstm_cell/bias/ApplyAdam	ApplyAdamq_next/rnn/lstm_cell/bias%q_next/q_next/rnn/lstm_cell/bias/Adam'q_next/q_next/rnn/lstm_cell/bias/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilonFq_next/gradients/q_next/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
use_nesterov( *
_output_shapes	
:
О
+q_next/Adam/update_q_next/weights/ApplyAdam	ApplyAdamq_next/weightsq_next/q_next/weights/Adamq_next/q_next/weights/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilon>q_next/gradients/q_next/MatMul_grad/tuple/control_dependency_1*
T0*!
_class
loc:@q_next/weights*
use_nesterov( *
_output_shapes
:	*
use_locking( 
Б
*q_next/Adam/update_q_next/biases/ApplyAdam	ApplyAdamq_next/biasesq_next/q_next/biases/Adamq_next/q_next/biases/Adam_1q_next/beta1_power/readq_next/beta2_power/readq_next/Adam/learning_rateq_next/Adam/beta1q_next/Adam/beta2q_next/Adam/epsilon;q_next/gradients/q_next/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0* 
_class
loc:@q_next/biases

q_next/Adam/mulMulq_next/beta1_power/readq_next/Adam/beta1+^q_next/Adam/update_q_next/biases/ApplyAdam/^q_next/Adam/update_q_next/conv1/bias/ApplyAdam1^q_next/Adam/update_q_next/conv1/kernel/ApplyAdam/^q_next/Adam/update_q_next/conv2/bias/ApplyAdam1^q_next/Adam/update_q_next/conv2/kernel/ApplyAdam7^q_next/Adam/update_q_next/rnn/lstm_cell/bias/ApplyAdam9^q_next/Adam/update_q_next/rnn/lstm_cell/kernel/ApplyAdam,^q_next/Adam/update_q_next/weights/ApplyAdam*
T0* 
_class
loc:@q_next/biases*
_output_shapes
: 
­
q_next/Adam/AssignAssignq_next/beta1_powerq_next/Adam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0* 
_class
loc:@q_next/biases

q_next/Adam/mul_1Mulq_next/beta2_power/readq_next/Adam/beta2+^q_next/Adam/update_q_next/biases/ApplyAdam/^q_next/Adam/update_q_next/conv1/bias/ApplyAdam1^q_next/Adam/update_q_next/conv1/kernel/ApplyAdam/^q_next/Adam/update_q_next/conv2/bias/ApplyAdam1^q_next/Adam/update_q_next/conv2/kernel/ApplyAdam7^q_next/Adam/update_q_next/rnn/lstm_cell/bias/ApplyAdam9^q_next/Adam/update_q_next/rnn/lstm_cell/kernel/ApplyAdam,^q_next/Adam/update_q_next/weights/ApplyAdam*
T0* 
_class
loc:@q_next/biases*
_output_shapes
: 
Б
q_next/Adam/Assign_1Assignq_next/beta2_powerq_next/Adam/mul_1*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: *
use_locking( 
ж
q_next/AdamNoOp^q_next/Adam/Assign^q_next/Adam/Assign_1+^q_next/Adam/update_q_next/biases/ApplyAdam/^q_next/Adam/update_q_next/conv1/bias/ApplyAdam1^q_next/Adam/update_q_next/conv1/kernel/ApplyAdam/^q_next/Adam/update_q_next/conv2/bias/ApplyAdam1^q_next/Adam/update_q_next/conv2/kernel/ApplyAdam7^q_next/Adam/update_q_next/rnn/lstm_cell/bias/ApplyAdam9^q_next/Adam/update_q_next/rnn/lstm_cell/kernel/ApplyAdam,^q_next/Adam/update_q_next/weights/ApplyAdam

init_1NoOp^q_eval/beta1_power/Assign^q_eval/beta2_power/Assign^q_eval/biases/Assign^q_eval/conv1/bias/Assign^q_eval/conv1/kernel/Assign^q_eval/conv2/bias/Assign^q_eval/conv2/kernel/Assign!^q_eval/q_eval/biases/Adam/Assign#^q_eval/q_eval/biases/Adam_1/Assign%^q_eval/q_eval/conv1/bias/Adam/Assign'^q_eval/q_eval/conv1/bias/Adam_1/Assign'^q_eval/q_eval/conv1/kernel/Adam/Assign)^q_eval/q_eval/conv1/kernel/Adam_1/Assign%^q_eval/q_eval/conv2/bias/Adam/Assign'^q_eval/q_eval/conv2/bias/Adam_1/Assign'^q_eval/q_eval/conv2/kernel/Adam/Assign)^q_eval/q_eval/conv2/kernel/Adam_1/Assign-^q_eval/q_eval/rnn/lstm_cell/bias/Adam/Assign/^q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Assign/^q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Assign1^q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Assign"^q_eval/q_eval/weights/Adam/Assign$^q_eval/q_eval/weights/Adam_1/Assign!^q_eval/rnn/lstm_cell/bias/Assign#^q_eval/rnn/lstm_cell/kernel/Assign^q_eval/weights/Assign^q_next/beta1_power/Assign^q_next/beta2_power/Assign^q_next/biases/Assign^q_next/conv1/bias/Assign^q_next/conv1/kernel/Assign^q_next/conv2/bias/Assign^q_next/conv2/kernel/Assign!^q_next/q_next/biases/Adam/Assign#^q_next/q_next/biases/Adam_1/Assign%^q_next/q_next/conv1/bias/Adam/Assign'^q_next/q_next/conv1/bias/Adam_1/Assign'^q_next/q_next/conv1/kernel/Adam/Assign)^q_next/q_next/conv1/kernel/Adam_1/Assign%^q_next/q_next/conv2/bias/Adam/Assign'^q_next/q_next/conv2/bias/Adam_1/Assign'^q_next/q_next/conv2/kernel/Adam/Assign)^q_next/q_next/conv2/kernel/Adam_1/Assign-^q_next/q_next/rnn/lstm_cell/bias/Adam/Assign/^q_next/q_next/rnn/lstm_cell/bias/Adam_1/Assign/^q_next/q_next/rnn/lstm_cell/kernel/Adam/Assign1^q_next/q_next/rnn/lstm_cell/kernel/Adam_1/Assign"^q_next/q_next/weights/Adam/Assign$^q_next/q_next/weights/Adam_1/Assign!^q_next/rnn/lstm_cell/bias/Assign#^q_next/rnn/lstm_cell/kernel/Assign^q_next/weights/Assign
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
О
save_1/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:4*я
valueхBт4Bq_eval/beta1_powerBq_eval/beta2_powerBq_eval/biasesBq_eval/conv1/biasBq_eval/conv1/kernelBq_eval/conv2/biasBq_eval/conv2/kernelBq_eval/q_eval/biases/AdamBq_eval/q_eval/biases/Adam_1Bq_eval/q_eval/conv1/bias/AdamBq_eval/q_eval/conv1/bias/Adam_1Bq_eval/q_eval/conv1/kernel/AdamB!q_eval/q_eval/conv1/kernel/Adam_1Bq_eval/q_eval/conv2/bias/AdamBq_eval/q_eval/conv2/bias/Adam_1Bq_eval/q_eval/conv2/kernel/AdamB!q_eval/q_eval/conv2/kernel/Adam_1B%q_eval/q_eval/rnn/lstm_cell/bias/AdamB'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1B'q_eval/q_eval/rnn/lstm_cell/kernel/AdamB)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1Bq_eval/q_eval/weights/AdamBq_eval/q_eval/weights/Adam_1Bq_eval/rnn/lstm_cell/biasBq_eval/rnn/lstm_cell/kernelBq_eval/weightsBq_next/beta1_powerBq_next/beta2_powerBq_next/biasesBq_next/conv1/biasBq_next/conv1/kernelBq_next/conv2/biasBq_next/conv2/kernelBq_next/q_next/biases/AdamBq_next/q_next/biases/Adam_1Bq_next/q_next/conv1/bias/AdamBq_next/q_next/conv1/bias/Adam_1Bq_next/q_next/conv1/kernel/AdamB!q_next/q_next/conv1/kernel/Adam_1Bq_next/q_next/conv2/bias/AdamBq_next/q_next/conv2/bias/Adam_1Bq_next/q_next/conv2/kernel/AdamB!q_next/q_next/conv2/kernel/Adam_1B%q_next/q_next/rnn/lstm_cell/bias/AdamB'q_next/q_next/rnn/lstm_cell/bias/Adam_1B'q_next/q_next/rnn/lstm_cell/kernel/AdamB)q_next/q_next/rnn/lstm_cell/kernel/Adam_1Bq_next/q_next/weights/AdamBq_next/q_next/weights/Adam_1Bq_next/rnn/lstm_cell/biasBq_next/rnn/lstm_cell/kernelBq_next/weights
Э
save_1/SaveV2/shape_and_slicesConst*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
џ
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesq_eval/beta1_powerq_eval/beta2_powerq_eval/biasesq_eval/conv1/biasq_eval/conv1/kernelq_eval/conv2/biasq_eval/conv2/kernelq_eval/q_eval/biases/Adamq_eval/q_eval/biases/Adam_1q_eval/q_eval/conv1/bias/Adamq_eval/q_eval/conv1/bias/Adam_1q_eval/q_eval/conv1/kernel/Adam!q_eval/q_eval/conv1/kernel/Adam_1q_eval/q_eval/conv2/bias/Adamq_eval/q_eval/conv2/bias/Adam_1q_eval/q_eval/conv2/kernel/Adam!q_eval/q_eval/conv2/kernel/Adam_1%q_eval/q_eval/rnn/lstm_cell/bias/Adam'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1'q_eval/q_eval/rnn/lstm_cell/kernel/Adam)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1q_eval/q_eval/weights/Adamq_eval/q_eval/weights/Adam_1q_eval/rnn/lstm_cell/biasq_eval/rnn/lstm_cell/kernelq_eval/weightsq_next/beta1_powerq_next/beta2_powerq_next/biasesq_next/conv1/biasq_next/conv1/kernelq_next/conv2/biasq_next/conv2/kernelq_next/q_next/biases/Adamq_next/q_next/biases/Adam_1q_next/q_next/conv1/bias/Adamq_next/q_next/conv1/bias/Adam_1q_next/q_next/conv1/kernel/Adam!q_next/q_next/conv1/kernel/Adam_1q_next/q_next/conv2/bias/Adamq_next/q_next/conv2/bias/Adam_1q_next/q_next/conv2/kernel/Adam!q_next/q_next/conv2/kernel/Adam_1%q_next/q_next/rnn/lstm_cell/bias/Adam'q_next/q_next/rnn/lstm_cell/bias/Adam_1'q_next/q_next/rnn/lstm_cell/kernel/Adam)q_next/q_next/rnn/lstm_cell/kernel/Adam_1q_next/q_next/weights/Adamq_next/q_next/weights/Adam_1q_next/rnn/lstm_cell/biasq_next/rnn/lstm_cell/kernelq_next/weights*B
dtypes8
624

save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
T0*
_class
loc:@save_1/Const*
_output_shapes
: 
а
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:4*я
valueхBт4Bq_eval/beta1_powerBq_eval/beta2_powerBq_eval/biasesBq_eval/conv1/biasBq_eval/conv1/kernelBq_eval/conv2/biasBq_eval/conv2/kernelBq_eval/q_eval/biases/AdamBq_eval/q_eval/biases/Adam_1Bq_eval/q_eval/conv1/bias/AdamBq_eval/q_eval/conv1/bias/Adam_1Bq_eval/q_eval/conv1/kernel/AdamB!q_eval/q_eval/conv1/kernel/Adam_1Bq_eval/q_eval/conv2/bias/AdamBq_eval/q_eval/conv2/bias/Adam_1Bq_eval/q_eval/conv2/kernel/AdamB!q_eval/q_eval/conv2/kernel/Adam_1B%q_eval/q_eval/rnn/lstm_cell/bias/AdamB'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1B'q_eval/q_eval/rnn/lstm_cell/kernel/AdamB)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1Bq_eval/q_eval/weights/AdamBq_eval/q_eval/weights/Adam_1Bq_eval/rnn/lstm_cell/biasBq_eval/rnn/lstm_cell/kernelBq_eval/weightsBq_next/beta1_powerBq_next/beta2_powerBq_next/biasesBq_next/conv1/biasBq_next/conv1/kernelBq_next/conv2/biasBq_next/conv2/kernelBq_next/q_next/biases/AdamBq_next/q_next/biases/Adam_1Bq_next/q_next/conv1/bias/AdamBq_next/q_next/conv1/bias/Adam_1Bq_next/q_next/conv1/kernel/AdamB!q_next/q_next/conv1/kernel/Adam_1Bq_next/q_next/conv2/bias/AdamBq_next/q_next/conv2/bias/Adam_1Bq_next/q_next/conv2/kernel/AdamB!q_next/q_next/conv2/kernel/Adam_1B%q_next/q_next/rnn/lstm_cell/bias/AdamB'q_next/q_next/rnn/lstm_cell/bias/Adam_1B'q_next/q_next/rnn/lstm_cell/kernel/AdamB)q_next/q_next/rnn/lstm_cell/kernel/Adam_1Bq_next/q_next/weights/AdamBq_next/q_next/weights/Adam_1Bq_next/rnn/lstm_cell/biasBq_next/rnn/lstm_cell/kernelBq_next/weights
п
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:4
Љ
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*ц
_output_shapesг
а::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624
Љ
save_1/AssignAssignq_eval/beta1_powersave_1/RestoreV2*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
: *
use_locking(
­
save_1/Assign_1Assignq_eval/beta2_powersave_1/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@q_eval/biases
Ќ
save_1/Assign_2Assignq_eval/biasessave_1/RestoreV2:2*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:*
use_locking(
Д
save_1/Assign_3Assignq_eval/conv1/biassave_1/RestoreV2:3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias
Ф
save_1/Assign_4Assignq_eval/conv1/kernelsave_1/RestoreV2:4*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
Д
save_1/Assign_5Assignq_eval/conv2/biassave_1/RestoreV2:5*
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: 
Ф
save_1/Assign_6Assignq_eval/conv2/kernelsave_1/RestoreV2:6*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel
И
save_1/Assign_7Assignq_eval/q_eval/biases/Adamsave_1/RestoreV2:7*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_eval/biases
К
save_1/Assign_8Assignq_eval/q_eval/biases/Adam_1save_1/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@q_eval/biases*
validate_shape(*
_output_shapes
:
Р
save_1/Assign_9Assignq_eval/q_eval/conv1/bias/Adamsave_1/RestoreV2:9*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:
Ф
save_1/Assign_10Assignq_eval/q_eval/conv1/bias/Adam_1save_1/RestoreV2:10*
use_locking(*
T0*$
_class
loc:@q_eval/conv1/bias*
validate_shape(*
_output_shapes
:
в
save_1/Assign_11Assignq_eval/q_eval/conv1/kernel/Adamsave_1/RestoreV2:11*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel
д
save_1/Assign_12Assign!q_eval/q_eval/conv1/kernel/Adam_1save_1/RestoreV2:12*
use_locking(*
T0*&
_class
loc:@q_eval/conv1/kernel*
validate_shape(*&
_output_shapes
:
Т
save_1/Assign_13Assignq_eval/q_eval/conv2/bias/Adamsave_1/RestoreV2:13*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias
Ф
save_1/Assign_14Assignq_eval/q_eval/conv2/bias/Adam_1save_1/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@q_eval/conv2/bias*
validate_shape(*
_output_shapes
: 
в
save_1/Assign_15Assignq_eval/q_eval/conv2/kernel/Adamsave_1/RestoreV2:15*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel
д
save_1/Assign_16Assign!q_eval/q_eval/conv2/kernel/Adam_1save_1/RestoreV2:16*
use_locking(*
T0*&
_class
loc:@q_eval/conv2/kernel*
validate_shape(*&
_output_shapes
: 
г
save_1/Assign_17Assign%q_eval/q_eval/rnn/lstm_cell/bias/Adamsave_1/RestoreV2:17*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
е
save_1/Assign_18Assign'q_eval/q_eval/rnn/lstm_cell/bias/Adam_1save_1/RestoreV2:18*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias
м
save_1/Assign_19Assign'q_eval/q_eval/rnn/lstm_cell/kernel/Adamsave_1/RestoreV2:19*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
о
save_1/Assign_20Assign)q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1save_1/RestoreV2:20*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
С
save_1/Assign_21Assignq_eval/q_eval/weights/Adamsave_1/RestoreV2:21*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
У
save_1/Assign_22Assignq_eval/q_eval/weights/Adam_1save_1/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@q_eval/weights*
validate_shape(*
_output_shapes
:	
Ч
save_1/Assign_23Assignq_eval/rnn/lstm_cell/biassave_1/RestoreV2:23*
T0*,
_class"
 loc:@q_eval/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
а
save_1/Assign_24Assignq_eval/rnn/lstm_cell/kernelsave_1/RestoreV2:24*
T0*.
_class$
" loc:@q_eval/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 *
use_locking(
Е
save_1/Assign_25Assignq_eval/weightssave_1/RestoreV2:25*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*!
_class
loc:@q_eval/weights
Џ
save_1/Assign_26Assignq_next/beta1_powersave_1/RestoreV2:26*
use_locking(*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
: 
Џ
save_1/Assign_27Assignq_next/beta2_powersave_1/RestoreV2:27*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@q_next/biases
Ў
save_1/Assign_28Assignq_next/biasessave_1/RestoreV2:28*
T0* 
_class
loc:@q_next/biases*
validate_shape(*
_output_shapes
:*
use_locking(
Ж
save_1/Assign_29Assignq_next/conv1/biassave_1/RestoreV2:29*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@q_next/conv1/bias
Ц
save_1/Assign_30Assignq_next/conv1/kernelsave_1/RestoreV2:30*
T0*&
_class
loc:@q_next/conv1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
Ж
save_1/Assign_31Assignq_next/conv2/biassave_1/RestoreV2:31*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias
Ц
save_1/Assign_32Assignq_next/conv2/kernelsave_1/RestoreV2:32*
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel*
validate_shape(*&
_output_shapes
: 
К
save_1/Assign_33Assignq_next/q_next/biases/Adamsave_1/RestoreV2:33*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_next/biases
М
save_1/Assign_34Assignq_next/q_next/biases/Adam_1save_1/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@q_next/biases
Т
save_1/Assign_35Assignq_next/q_next/conv1/bias/Adamsave_1/RestoreV2:35*
T0*$
_class
loc:@q_next/conv1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ф
save_1/Assign_36Assignq_next/q_next/conv1/bias/Adam_1save_1/RestoreV2:36*
T0*$
_class
loc:@q_next/conv1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
в
save_1/Assign_37Assignq_next/q_next/conv1/kernel/Adamsave_1/RestoreV2:37*
use_locking(*
T0*&
_class
loc:@q_next/conv1/kernel*
validate_shape(*&
_output_shapes
:
д
save_1/Assign_38Assign!q_next/q_next/conv1/kernel/Adam_1save_1/RestoreV2:38*
use_locking(*
T0*&
_class
loc:@q_next/conv1/kernel*
validate_shape(*&
_output_shapes
:
Т
save_1/Assign_39Assignq_next/q_next/conv2/bias/Adamsave_1/RestoreV2:39*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias
Ф
save_1/Assign_40Assignq_next/q_next/conv2/bias/Adam_1save_1/RestoreV2:40*
use_locking(*
T0*$
_class
loc:@q_next/conv2/bias*
validate_shape(*
_output_shapes
: 
в
save_1/Assign_41Assignq_next/q_next/conv2/kernel/Adamsave_1/RestoreV2:41*
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel*
validate_shape(*&
_output_shapes
: 
д
save_1/Assign_42Assign!q_next/q_next/conv2/kernel/Adam_1save_1/RestoreV2:42*
use_locking(*
T0*&
_class
loc:@q_next/conv2/kernel*
validate_shape(*&
_output_shapes
: 
г
save_1/Assign_43Assign%q_next/q_next/rnn/lstm_cell/bias/Adamsave_1/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:
е
save_1/Assign_44Assign'q_next/q_next/rnn/lstm_cell/bias/Adam_1save_1/RestoreV2:44*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
м
save_1/Assign_45Assign'q_next/q_next/rnn/lstm_cell/kernel/Adamsave_1/RestoreV2:45*
validate_shape(* 
_output_shapes
:
 *
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel
о
save_1/Assign_46Assign)q_next/q_next/rnn/lstm_cell/kernel/Adam_1save_1/RestoreV2:46*
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
С
save_1/Assign_47Assignq_next/q_next/weights/Adamsave_1/RestoreV2:47*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*!
_class
loc:@q_next/weights
У
save_1/Assign_48Assignq_next/q_next/weights/Adam_1save_1/RestoreV2:48*
use_locking(*
T0*!
_class
loc:@q_next/weights*
validate_shape(*
_output_shapes
:	
Ч
save_1/Assign_49Assignq_next/rnn/lstm_cell/biassave_1/RestoreV2:49*
T0*,
_class"
 loc:@q_next/rnn/lstm_cell/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
а
save_1/Assign_50Assignq_next/rnn/lstm_cell/kernelsave_1/RestoreV2:50*
use_locking(*
T0*.
_class$
" loc:@q_next/rnn/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
 
Е
save_1/Assign_51Assignq_next/weightssave_1/RestoreV2:51*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*!
_class
loc:@q_next/weights
ъ
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
М
Merge_1/MergeSummaryMergeSummaryq_eval/Reward/Time_step_1#q_eval/TotalWaitingTime/Time_step_1q_eval/TotalDelay/Time_step_1q_eval/Q_valueq_eval/Loss.q_eval/q_eval/conv1/kernel/summaries/histogram,q_eval/q_eval/conv1/bias/summaries/histogram.q_eval/q_eval/conv2/kernel/summaries/histogram,q_eval/q_eval/conv2/bias/summaries/histogram6q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram4q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram)q_eval/q_eval/weights/summaries/histogram(q_eval/q_eval/biases/summaries/histogramq_next/Reward/Time_step_1#q_next/TotalWaitingTime/Time_step_1q_next/TotalDelay/Time_step_1q_next/Q_valueq_next/Loss*
N*
_output_shapes
: ""
trainable_variablesѓ№

q_eval/conv1/kernel:0q_eval/conv1/kernel/Assignq_eval/conv1/kernel/read:022q_eval/conv1/kernel/Initializer/truncated_normal:0
p
q_eval/conv1/bias:0q_eval/conv1/bias/Assignq_eval/conv1/bias/read:02%q_eval/conv1/bias/Initializer/Const:0

q_eval/conv2/kernel:0q_eval/conv2/kernel/Assignq_eval/conv2/kernel/read:022q_eval/conv2/kernel/Initializer/truncated_normal:0
p
q_eval/conv2/bias:0q_eval/conv2/bias/Assignq_eval/conv2/bias/read:02%q_eval/conv2/bias/Initializer/Const:0
Ё
q_eval/rnn/lstm_cell/kernel:0"q_eval/rnn/lstm_cell/kernel/Assign"q_eval/rnn/lstm_cell/kernel/read:028q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform:0

q_eval/rnn/lstm_cell/bias:0 q_eval/rnn/lstm_cell/bias/Assign q_eval/rnn/lstm_cell/bias/read:02-q_eval/rnn/lstm_cell/bias/Initializer/zeros:0
m
q_eval/weights:0q_eval/weights/Assignq_eval/weights/read:02+q_eval/weights/Initializer/random_uniform:0
`
q_eval/biases:0q_eval/biases/Assignq_eval/biases/read:02!q_eval/biases/Initializer/Const:0

q_next/conv1/kernel:0q_next/conv1/kernel/Assignq_next/conv1/kernel/read:022q_next/conv1/kernel/Initializer/truncated_normal:0
p
q_next/conv1/bias:0q_next/conv1/bias/Assignq_next/conv1/bias/read:02%q_next/conv1/bias/Initializer/Const:0

q_next/conv2/kernel:0q_next/conv2/kernel/Assignq_next/conv2/kernel/read:022q_next/conv2/kernel/Initializer/truncated_normal:0
p
q_next/conv2/bias:0q_next/conv2/bias/Assignq_next/conv2/bias/read:02%q_next/conv2/bias/Initializer/Const:0
Ё
q_next/rnn/lstm_cell/kernel:0"q_next/rnn/lstm_cell/kernel/Assign"q_next/rnn/lstm_cell/kernel/read:028q_next/rnn/lstm_cell/kernel/Initializer/random_uniform:0

q_next/rnn/lstm_cell/bias:0 q_next/rnn/lstm_cell/bias/Assign q_next/rnn/lstm_cell/bias/read:02-q_next/rnn/lstm_cell/bias/Initializer/zeros:0
m
q_next/weights:0q_next/weights/Assignq_next/weights/read:02+q_next/weights/Initializer/random_uniform:0
`
q_next/biases:0q_next/biases/Assignq_next/biases/read:02!q_next/biases/Initializer/Const:0"Ќ
	summaries

q_eval/Reward/Time_step_1:0
%q_eval/TotalWaitingTime/Time_step_1:0
q_eval/TotalDelay/Time_step_1:0
q_eval/Q_value:0
q_eval/Loss:0
0q_eval/q_eval/conv1/kernel/summaries/histogram:0
.q_eval/q_eval/conv1/bias/summaries/histogram:0
0q_eval/q_eval/conv2/kernel/summaries/histogram:0
.q_eval/q_eval/conv2/bias/summaries/histogram:0
8q_eval/q_eval/rnn/lstm_cell/kernel/summaries/histogram:0
6q_eval/q_eval/rnn/lstm_cell/bias/summaries/histogram:0
+q_eval/q_eval/weights/summaries/histogram:0
*q_eval/q_eval/biases/summaries/histogram:0
q_next/Reward/Time_step_1:0
%q_next/TotalWaitingTime/Time_step_1:0
q_next/TotalDelay/Time_step_1:0
q_next/Q_value:0
q_next/Loss:0"?
	variables??

q_eval/conv1/kernel:0q_eval/conv1/kernel/Assignq_eval/conv1/kernel/read:022q_eval/conv1/kernel/Initializer/truncated_normal:0
p
q_eval/conv1/bias:0q_eval/conv1/bias/Assignq_eval/conv1/bias/read:02%q_eval/conv1/bias/Initializer/Const:0

q_eval/conv2/kernel:0q_eval/conv2/kernel/Assignq_eval/conv2/kernel/read:022q_eval/conv2/kernel/Initializer/truncated_normal:0
p
q_eval/conv2/bias:0q_eval/conv2/bias/Assignq_eval/conv2/bias/read:02%q_eval/conv2/bias/Initializer/Const:0
Ё
q_eval/rnn/lstm_cell/kernel:0"q_eval/rnn/lstm_cell/kernel/Assign"q_eval/rnn/lstm_cell/kernel/read:028q_eval/rnn/lstm_cell/kernel/Initializer/random_uniform:0

q_eval/rnn/lstm_cell/bias:0 q_eval/rnn/lstm_cell/bias/Assign q_eval/rnn/lstm_cell/bias/read:02-q_eval/rnn/lstm_cell/bias/Initializer/zeros:0
m
q_eval/weights:0q_eval/weights/Assignq_eval/weights/read:02+q_eval/weights/Initializer/random_uniform:0
`
q_eval/biases:0q_eval/biases/Assignq_eval/biases/read:02!q_eval/biases/Initializer/Const:0
p
q_eval/beta1_power:0q_eval/beta1_power/Assignq_eval/beta1_power/read:02"q_eval/beta1_power/initial_value:0
p
q_eval/beta2_power:0q_eval/beta2_power/Assignq_eval/beta2_power/read:02"q_eval/beta2_power/initial_value:0
Ј
!q_eval/q_eval/conv1/kernel/Adam:0&q_eval/q_eval/conv1/kernel/Adam/Assign&q_eval/q_eval/conv1/kernel/Adam/read:023q_eval/q_eval/conv1/kernel/Adam/Initializer/zeros:0
А
#q_eval/q_eval/conv1/kernel/Adam_1:0(q_eval/q_eval/conv1/kernel/Adam_1/Assign(q_eval/q_eval/conv1/kernel/Adam_1/read:025q_eval/q_eval/conv1/kernel/Adam_1/Initializer/zeros:0
 
q_eval/q_eval/conv1/bias/Adam:0$q_eval/q_eval/conv1/bias/Adam/Assign$q_eval/q_eval/conv1/bias/Adam/read:021q_eval/q_eval/conv1/bias/Adam/Initializer/zeros:0
Ј
!q_eval/q_eval/conv1/bias/Adam_1:0&q_eval/q_eval/conv1/bias/Adam_1/Assign&q_eval/q_eval/conv1/bias/Adam_1/read:023q_eval/q_eval/conv1/bias/Adam_1/Initializer/zeros:0
Ј
!q_eval/q_eval/conv2/kernel/Adam:0&q_eval/q_eval/conv2/kernel/Adam/Assign&q_eval/q_eval/conv2/kernel/Adam/read:023q_eval/q_eval/conv2/kernel/Adam/Initializer/zeros:0
А
#q_eval/q_eval/conv2/kernel/Adam_1:0(q_eval/q_eval/conv2/kernel/Adam_1/Assign(q_eval/q_eval/conv2/kernel/Adam_1/read:025q_eval/q_eval/conv2/kernel/Adam_1/Initializer/zeros:0
 
q_eval/q_eval/conv2/bias/Adam:0$q_eval/q_eval/conv2/bias/Adam/Assign$q_eval/q_eval/conv2/bias/Adam/read:021q_eval/q_eval/conv2/bias/Adam/Initializer/zeros:0
Ј
!q_eval/q_eval/conv2/bias/Adam_1:0&q_eval/q_eval/conv2/bias/Adam_1/Assign&q_eval/q_eval/conv2/bias/Adam_1/read:023q_eval/q_eval/conv2/bias/Adam_1/Initializer/zeros:0
Ш
)q_eval/q_eval/rnn/lstm_cell/kernel/Adam:0.q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Assign.q_eval/q_eval/rnn/lstm_cell/kernel/Adam/read:02;q_eval/q_eval/rnn/lstm_cell/kernel/Adam/Initializer/zeros:0
а
+q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1:00q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Assign0q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/read:02=q_eval/q_eval/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros:0
Р
'q_eval/q_eval/rnn/lstm_cell/bias/Adam:0,q_eval/q_eval/rnn/lstm_cell/bias/Adam/Assign,q_eval/q_eval/rnn/lstm_cell/bias/Adam/read:029q_eval/q_eval/rnn/lstm_cell/bias/Adam/Initializer/zeros:0
Ш
)q_eval/q_eval/rnn/lstm_cell/bias/Adam_1:0.q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Assign.q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/read:02;q_eval/q_eval/rnn/lstm_cell/bias/Adam_1/Initializer/zeros:0

q_eval/q_eval/weights/Adam:0!q_eval/q_eval/weights/Adam/Assign!q_eval/q_eval/weights/Adam/read:02.q_eval/q_eval/weights/Adam/Initializer/zeros:0

q_eval/q_eval/weights/Adam_1:0#q_eval/q_eval/weights/Adam_1/Assign#q_eval/q_eval/weights/Adam_1/read:020q_eval/q_eval/weights/Adam_1/Initializer/zeros:0

q_eval/q_eval/biases/Adam:0 q_eval/q_eval/biases/Adam/Assign q_eval/q_eval/biases/Adam/read:02-q_eval/q_eval/biases/Adam/Initializer/zeros:0

q_eval/q_eval/biases/Adam_1:0"q_eval/q_eval/biases/Adam_1/Assign"q_eval/q_eval/biases/Adam_1/read:02/q_eval/q_eval/biases/Adam_1/Initializer/zeros:0

q_next/conv1/kernel:0q_next/conv1/kernel/Assignq_next/conv1/kernel/read:022q_next/conv1/kernel/Initializer/truncated_normal:0
p
q_next/conv1/bias:0q_next/conv1/bias/Assignq_next/conv1/bias/read:02%q_next/conv1/bias/Initializer/Const:0

q_next/conv2/kernel:0q_next/conv2/kernel/Assignq_next/conv2/kernel/read:022q_next/conv2/kernel/Initializer/truncated_normal:0
p
q_next/conv2/bias:0q_next/conv2/bias/Assignq_next/conv2/bias/read:02%q_next/conv2/bias/Initializer/Const:0
Ё
q_next/rnn/lstm_cell/kernel:0"q_next/rnn/lstm_cell/kernel/Assign"q_next/rnn/lstm_cell/kernel/read:028q_next/rnn/lstm_cell/kernel/Initializer/random_uniform:0

q_next/rnn/lstm_cell/bias:0 q_next/rnn/lstm_cell/bias/Assign q_next/rnn/lstm_cell/bias/read:02-q_next/rnn/lstm_cell/bias/Initializer/zeros:0
m
q_next/weights:0q_next/weights/Assignq_next/weights/read:02+q_next/weights/Initializer/random_uniform:0
`
q_next/biases:0q_next/biases/Assignq_next/biases/read:02!q_next/biases/Initializer/Const:0
p
q_next/beta1_power:0q_next/beta1_power/Assignq_next/beta1_power/read:02"q_next/beta1_power/initial_value:0
p
q_next/beta2_power:0q_next/beta2_power/Assignq_next/beta2_power/read:02"q_next/beta2_power/initial_value:0
Ј
!q_next/q_next/conv1/kernel/Adam:0&q_next/q_next/conv1/kernel/Adam/Assign&q_next/q_next/conv1/kernel/Adam/read:023q_next/q_next/conv1/kernel/Adam/Initializer/zeros:0
А
#q_next/q_next/conv1/kernel/Adam_1:0(q_next/q_next/conv1/kernel/Adam_1/Assign(q_next/q_next/conv1/kernel/Adam_1/read:025q_next/q_next/conv1/kernel/Adam_1/Initializer/zeros:0
 
q_next/q_next/conv1/bias/Adam:0$q_next/q_next/conv1/bias/Adam/Assign$q_next/q_next/conv1/bias/Adam/read:021q_next/q_next/conv1/bias/Adam/Initializer/zeros:0
Ј
!q_next/q_next/conv1/bias/Adam_1:0&q_next/q_next/conv1/bias/Adam_1/Assign&q_next/q_next/conv1/bias/Adam_1/read:023q_next/q_next/conv1/bias/Adam_1/Initializer/zeros:0
Ј
!q_next/q_next/conv2/kernel/Adam:0&q_next/q_next/conv2/kernel/Adam/Assign&q_next/q_next/conv2/kernel/Adam/read:023q_next/q_next/conv2/kernel/Adam/Initializer/zeros:0
А
#q_next/q_next/conv2/kernel/Adam_1:0(q_next/q_next/conv2/kernel/Adam_1/Assign(q_next/q_next/conv2/kernel/Adam_1/read:025q_next/q_next/conv2/kernel/Adam_1/Initializer/zeros:0
 
q_next/q_next/conv2/bias/Adam:0$q_next/q_next/conv2/bias/Adam/Assign$q_next/q_next/conv2/bias/Adam/read:021q_next/q_next/conv2/bias/Adam/Initializer/zeros:0
Ј
!q_next/q_next/conv2/bias/Adam_1:0&q_next/q_next/conv2/bias/Adam_1/Assign&q_next/q_next/conv2/bias/Adam_1/read:023q_next/q_next/conv2/bias/Adam_1/Initializer/zeros:0
Ш
)q_next/q_next/rnn/lstm_cell/kernel/Adam:0.q_next/q_next/rnn/lstm_cell/kernel/Adam/Assign.q_next/q_next/rnn/lstm_cell/kernel/Adam/read:02;q_next/q_next/rnn/lstm_cell/kernel/Adam/Initializer/zeros:0
а
+q_next/q_next/rnn/lstm_cell/kernel/Adam_1:00q_next/q_next/rnn/lstm_cell/kernel/Adam_1/Assign0q_next/q_next/rnn/lstm_cell/kernel/Adam_1/read:02=q_next/q_next/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros:0
Р
'q_next/q_next/rnn/lstm_cell/bias/Adam:0,q_next/q_next/rnn/lstm_cell/bias/Adam/Assign,q_next/q_next/rnn/lstm_cell/bias/Adam/read:029q_next/q_next/rnn/lstm_cell/bias/Adam/Initializer/zeros:0
Ш
)q_next/q_next/rnn/lstm_cell/bias/Adam_1:0.q_next/q_next/rnn/lstm_cell/bias/Adam_1/Assign.q_next/q_next/rnn/lstm_cell/bias/Adam_1/read:02;q_next/q_next/rnn/lstm_cell/bias/Adam_1/Initializer/zeros:0

q_next/q_next/weights/Adam:0!q_next/q_next/weights/Adam/Assign!q_next/q_next/weights/Adam/read:02.q_next/q_next/weights/Adam/Initializer/zeros:0

q_next/q_next/weights/Adam_1:0#q_next/q_next/weights/Adam_1/Assign#q_next/q_next/weights/Adam_1/read:020q_next/q_next/weights/Adam_1/Initializer/zeros:0

q_next/q_next/biases/Adam:0 q_next/q_next/biases/Adam/Assign q_next/q_next/biases/Adam/read:02-q_next/q_next/biases/Adam/Initializer/zeros:0

q_next/q_next/biases/Adam_1:0"q_next/q_next/biases/Adam_1/Assign"q_next/q_next/biases/Adam_1/read:02/q_next/q_next/biases/Adam_1/Initializer/zeros:0"чЭ
while_contextдЭаЭ
хf
q_eval/rnn/while/while_context *q_eval/rnn/while/LoopCond:02q_eval/rnn/while/Merge:0:q_eval/rnn/while/Identity:0Bq_eval/rnn/while/Exit:0Bq_eval/rnn/while/Exit_1:0Bq_eval/rnn/while/Exit_2:0Bq_eval/rnn/while/Exit_3:0Bq_eval/rnn/while/Exit_4:0Bq_eval/gradients/f_count_2:0Jc
q_eval/gradients/Add/y:0
q_eval/gradients/Add:0
q_eval/gradients/Merge:0
q_eval/gradients/Merge:1
 q_eval/gradients/NextIteration:0
q_eval/gradients/Switch:0
q_eval/gradients/Switch:1
q_eval/gradients/f_count:0
q_eval/gradients/f_count_1:0
q_eval/gradients/f_count_2:0
>q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/Enter:0
Dq_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/StackPushV2:0
>q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/f_acc:0
Bq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/Enter:0
Hq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/StackPushV2:0
Bq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/f_acc:0
Bq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/Enter:0
Hq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/StackPushV2:0
Bq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/f_acc:0
dq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
jq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
dq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
Nq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
\q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
>q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape:0
@q_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/Shape_1:0
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
<q_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/Shape:0
?q_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/Shape:0
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/Enter:0
Lq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc:0
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
\q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/Enter:0
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Jq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape:0
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Shape_1:0
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
\q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/Enter:0
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
Jq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape:0
@q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Shape_1:0
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
Xq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
Zq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/Enter:0
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc:0
<q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape:0
>q_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Shape_1:0
q_eval/rnn/CheckSeqLen:0
q_eval/rnn/Minimum:0
q_eval/rnn/TensorArray:0
Gq_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
q_eval/rnn/TensorArray_1:0
 q_eval/rnn/lstm_cell/bias/read:0
"q_eval/rnn/lstm_cell/kernel/read:0
q_eval/rnn/strided_slice_1:0
q_eval/rnn/while/Enter:0
q_eval/rnn/while/Enter_1:0
q_eval/rnn/while/Enter_2:0
q_eval/rnn/while/Enter_3:0
q_eval/rnn/while/Enter_4:0
q_eval/rnn/while/Exit:0
q_eval/rnn/while/Exit_1:0
q_eval/rnn/while/Exit_2:0
q_eval/rnn/while/Exit_3:0
q_eval/rnn/while/Exit_4:0
%q_eval/rnn/while/GreaterEqual/Enter:0
q_eval/rnn/while/GreaterEqual:0
q_eval/rnn/while/Identity:0
q_eval/rnn/while/Identity_1:0
q_eval/rnn/while/Identity_2:0
q_eval/rnn/while/Identity_3:0
q_eval/rnn/while/Identity_4:0
q_eval/rnn/while/Less/Enter:0
q_eval/rnn/while/Less:0
q_eval/rnn/while/Less_1/Enter:0
q_eval/rnn/while/Less_1:0
q_eval/rnn/while/LogicalAnd:0
q_eval/rnn/while/LoopCond:0
q_eval/rnn/while/Merge:0
q_eval/rnn/while/Merge:1
q_eval/rnn/while/Merge_1:0
q_eval/rnn/while/Merge_1:1
q_eval/rnn/while/Merge_2:0
q_eval/rnn/while/Merge_2:1
q_eval/rnn/while/Merge_3:0
q_eval/rnn/while/Merge_3:1
q_eval/rnn/while/Merge_4:0
q_eval/rnn/while/Merge_4:1
 q_eval/rnn/while/NextIteration:0
"q_eval/rnn/while/NextIteration_1:0
"q_eval/rnn/while/NextIteration_2:0
"q_eval/rnn/while/NextIteration_3:0
"q_eval/rnn/while/NextIteration_4:0
q_eval/rnn/while/Select/Enter:0
q_eval/rnn/while/Select:0
q_eval/rnn/while/Select_1:0
q_eval/rnn/while/Select_2:0
q_eval/rnn/while/Switch:0
q_eval/rnn/while/Switch:1
q_eval/rnn/while/Switch_1:0
q_eval/rnn/while/Switch_1:1
q_eval/rnn/while/Switch_2:0
q_eval/rnn/while/Switch_2:1
q_eval/rnn/while/Switch_3:0
q_eval/rnn/while/Switch_3:1
q_eval/rnn/while/Switch_4:0
q_eval/rnn/while/Switch_4:1
*q_eval/rnn/while/TensorArrayReadV3/Enter:0
,q_eval/rnn/while/TensorArrayReadV3/Enter_1:0
$q_eval/rnn/while/TensorArrayReadV3:0
<q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
6q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
q_eval/rnn/while/add/y:0
q_eval/rnn/while/add:0
q_eval/rnn/while/add_1/y:0
q_eval/rnn/while/add_1:0
*q_eval/rnn/while/lstm_cell/BiasAdd/Enter:0
$q_eval/rnn/while/lstm_cell/BiasAdd:0
"q_eval/rnn/while/lstm_cell/Const:0
)q_eval/rnn/while/lstm_cell/MatMul/Enter:0
#q_eval/rnn/while/lstm_cell/MatMul:0
$q_eval/rnn/while/lstm_cell/Sigmoid:0
&q_eval/rnn/while/lstm_cell/Sigmoid_1:0
&q_eval/rnn/while/lstm_cell/Sigmoid_2:0
!q_eval/rnn/while/lstm_cell/Tanh:0
#q_eval/rnn/while/lstm_cell/Tanh_1:0
"q_eval/rnn/while/lstm_cell/add/y:0
 q_eval/rnn/while/lstm_cell/add:0
"q_eval/rnn/while/lstm_cell/add_1:0
(q_eval/rnn/while/lstm_cell/concat/axis:0
#q_eval/rnn/while/lstm_cell/concat:0
 q_eval/rnn/while/lstm_cell/mul:0
"q_eval/rnn/while/lstm_cell/mul_1:0
"q_eval/rnn/while/lstm_cell/mul_2:0
,q_eval/rnn/while/lstm_cell/split/split_dim:0
"q_eval/rnn/while/lstm_cell/split:0
"q_eval/rnn/while/lstm_cell/split:1
"q_eval/rnn/while/lstm_cell/split:2
"q_eval/rnn/while/lstm_cell/split:3
q_eval/rnn/zeros:0Ќ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc:0Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul/Enter:0
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc:0Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/Mul_1/Enter:0А
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0=
q_eval/rnn/strided_slice_1:0q_eval/rnn/while/Less/Enter:0
Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc:0Bq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul/Enter:0O
"q_eval/rnn/lstm_cell/kernel/read:0)q_eval/rnn/while/lstm_cell/MatMul/Enter:0
Hq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Hq_eval/gradients/q_eval/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0Ј
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0Rq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0H
q_eval/rnn/TensorArray_1:0*q_eval/rnn/while/TensorArrayReadV3/Enter:0
Bq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/f_acc:0Bq_eval/gradients/q_eval/rnn/while/Select_1_grad/zeros_like/Enter:0X
q_eval/rnn/TensorArray:0<q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
>q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/f_acc:0>q_eval/gradients/q_eval/rnn/while/Select_1_grad/Select/Enter:0Ќ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0Tq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:07
q_eval/rnn/Minimum:0q_eval/rnn/while/Less_1/Enter:0N
 q_eval/rnn/lstm_cell/bias/read:0*q_eval/rnn/while/lstm_cell/BiasAdd/Enter:0
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Dq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
Dq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Dq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter:0Ь
dq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0dq_eval/gradients/q_eval/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:05
q_eval/rnn/zeros:0q_eval/rnn/while/Select/Enter:0
Fq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc:0Fq_eval/gradients/q_eval/rnn/while/lstm_cell/concat_grad/ShapeN/Enter:0А
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0Vq_eval/gradients/q_eval/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0Ј
Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0Rq_eval/gradients/q_eval/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0A
q_eval/rnn/CheckSeqLen:0%q_eval/rnn/while/GreaterEqual/Enter:0Ќ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0Ќ
Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0Tq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0А
Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0Vq_eval/gradients/q_eval/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0w
Gq_eval/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0,q_eval/rnn/while/TensorArrayReadV3/Enter_1:0
Bq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/f_acc:0Bq_eval/gradients/q_eval/rnn/while/Select_2_grad/zeros_like/Enter:0Rq_eval/rnn/while/Enter:0Rq_eval/rnn/while/Enter_1:0Rq_eval/rnn/while/Enter_2:0Rq_eval/rnn/while/Enter_3:0Rq_eval/rnn/while/Enter_4:0Rq_eval/gradients/f_count_1:0Zq_eval/rnn/strided_slice_1:0
хf
q_next/rnn/while/while_context *q_next/rnn/while/LoopCond:02q_next/rnn/while/Merge:0:q_next/rnn/while/Identity:0Bq_next/rnn/while/Exit:0Bq_next/rnn/while/Exit_1:0Bq_next/rnn/while/Exit_2:0Bq_next/rnn/while/Exit_3:0Bq_next/rnn/while/Exit_4:0Bq_next/gradients/f_count_2:0Jc
q_next/gradients/Add/y:0
q_next/gradients/Add:0
q_next/gradients/Merge:0
q_next/gradients/Merge:1
 q_next/gradients/NextIteration:0
q_next/gradients/Switch:0
q_next/gradients/Switch:1
q_next/gradients/f_count:0
q_next/gradients/f_count_1:0
q_next/gradients/f_count_2:0
>q_next/gradients/q_next/rnn/while/Select_1_grad/Select/Enter:0
Dq_next/gradients/q_next/rnn/while/Select_1_grad/Select/StackPushV2:0
>q_next/gradients/q_next/rnn/while/Select_1_grad/Select/f_acc:0
Bq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/Enter:0
Hq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/StackPushV2:0
Bq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/f_acc:0
Bq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/Enter:0
Hq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/StackPushV2:0
Bq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/f_acc:0
dq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
jq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
dq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Hq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
Nq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Hq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
Vq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
Zq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
\q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
Vq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
>q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape:0
@q_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/Shape_1:0
Rq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
Xq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
Rq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
<q_next/gradients/q_next/rnn/while/lstm_cell/add_grad/Shape:0
?q_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/Shape:0
Fq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/Enter:0
Lq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Fq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc:0
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
Zq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
\q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/Enter:0
Hq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Dq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Jq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Dq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
>q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape:0
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Shape_1:0
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
Zq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
\q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/Enter:0
Hq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Dq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
Jq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Dq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
>q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape:0
@q_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Shape_1:0
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
Xq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
Zq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/Enter:0
Hq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc:0
<q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape:0
>q_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Shape_1:0
q_next/rnn/CheckSeqLen:0
q_next/rnn/Minimum:0
q_next/rnn/TensorArray:0
Gq_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
q_next/rnn/TensorArray_1:0
 q_next/rnn/lstm_cell/bias/read:0
"q_next/rnn/lstm_cell/kernel/read:0
q_next/rnn/strided_slice_1:0
q_next/rnn/while/Enter:0
q_next/rnn/while/Enter_1:0
q_next/rnn/while/Enter_2:0
q_next/rnn/while/Enter_3:0
q_next/rnn/while/Enter_4:0
q_next/rnn/while/Exit:0
q_next/rnn/while/Exit_1:0
q_next/rnn/while/Exit_2:0
q_next/rnn/while/Exit_3:0
q_next/rnn/while/Exit_4:0
%q_next/rnn/while/GreaterEqual/Enter:0
q_next/rnn/while/GreaterEqual:0
q_next/rnn/while/Identity:0
q_next/rnn/while/Identity_1:0
q_next/rnn/while/Identity_2:0
q_next/rnn/while/Identity_3:0
q_next/rnn/while/Identity_4:0
q_next/rnn/while/Less/Enter:0
q_next/rnn/while/Less:0
q_next/rnn/while/Less_1/Enter:0
q_next/rnn/while/Less_1:0
q_next/rnn/while/LogicalAnd:0
q_next/rnn/while/LoopCond:0
q_next/rnn/while/Merge:0
q_next/rnn/while/Merge:1
q_next/rnn/while/Merge_1:0
q_next/rnn/while/Merge_1:1
q_next/rnn/while/Merge_2:0
q_next/rnn/while/Merge_2:1
q_next/rnn/while/Merge_3:0
q_next/rnn/while/Merge_3:1
q_next/rnn/while/Merge_4:0
q_next/rnn/while/Merge_4:1
 q_next/rnn/while/NextIteration:0
"q_next/rnn/while/NextIteration_1:0
"q_next/rnn/while/NextIteration_2:0
"q_next/rnn/while/NextIteration_3:0
"q_next/rnn/while/NextIteration_4:0
q_next/rnn/while/Select/Enter:0
q_next/rnn/while/Select:0
q_next/rnn/while/Select_1:0
q_next/rnn/while/Select_2:0
q_next/rnn/while/Switch:0
q_next/rnn/while/Switch:1
q_next/rnn/while/Switch_1:0
q_next/rnn/while/Switch_1:1
q_next/rnn/while/Switch_2:0
q_next/rnn/while/Switch_2:1
q_next/rnn/while/Switch_3:0
q_next/rnn/while/Switch_3:1
q_next/rnn/while/Switch_4:0
q_next/rnn/while/Switch_4:1
*q_next/rnn/while/TensorArrayReadV3/Enter:0
,q_next/rnn/while/TensorArrayReadV3/Enter_1:0
$q_next/rnn/while/TensorArrayReadV3:0
<q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
6q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
q_next/rnn/while/add/y:0
q_next/rnn/while/add:0
q_next/rnn/while/add_1/y:0
q_next/rnn/while/add_1:0
*q_next/rnn/while/lstm_cell/BiasAdd/Enter:0
$q_next/rnn/while/lstm_cell/BiasAdd:0
"q_next/rnn/while/lstm_cell/Const:0
)q_next/rnn/while/lstm_cell/MatMul/Enter:0
#q_next/rnn/while/lstm_cell/MatMul:0
$q_next/rnn/while/lstm_cell/Sigmoid:0
&q_next/rnn/while/lstm_cell/Sigmoid_1:0
&q_next/rnn/while/lstm_cell/Sigmoid_2:0
!q_next/rnn/while/lstm_cell/Tanh:0
#q_next/rnn/while/lstm_cell/Tanh_1:0
"q_next/rnn/while/lstm_cell/add/y:0
 q_next/rnn/while/lstm_cell/add:0
"q_next/rnn/while/lstm_cell/add_1:0
(q_next/rnn/while/lstm_cell/concat/axis:0
#q_next/rnn/while/lstm_cell/concat:0
 q_next/rnn/while/lstm_cell/mul:0
"q_next/rnn/while/lstm_cell/mul_1:0
"q_next/rnn/while/lstm_cell/mul_2:0
,q_next/rnn/while/lstm_cell/split/split_dim:0
"q_next/rnn/while/lstm_cell/split:0
"q_next/rnn/while/lstm_cell/split:1
"q_next/rnn/while/lstm_cell/split:2
"q_next/rnn/while/lstm_cell/split:3
q_next/rnn/zeros:0X
q_next/rnn/TensorArray:0<q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Dq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Dq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Bq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/f_acc:0Bq_next/gradients/q_next/rnn/while/Select_1_grad/zeros_like/Enter:0А
Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0Ь
dq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0dq_next/gradients/q_next/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
Hq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Hq_next/gradients/q_next/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0А
Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0Vq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0Ќ
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0Ќ
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0w
Gq_next/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0,q_next/rnn/while/TensorArrayReadV3/Enter_1:0Ј
Rq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0Rq_next/gradients/q_next/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0A
q_next/rnn/CheckSeqLen:0%q_next/rnn/while/GreaterEqual/Enter:0Ќ
Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0Tq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:05
q_next/rnn/zeros:0q_next/rnn/while/Select/Enter:0
>q_next/gradients/q_next/rnn/while/Select_1_grad/Select/f_acc:0>q_next/gradients/q_next/rnn/while/Select_1_grad/Select/Enter:0
Bq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/f_acc:0Bq_next/gradients/q_next/rnn/while/Select_2_grad/zeros_like/Enter:0H
q_next/rnn/TensorArray_1:0*q_next/rnn/while/TensorArrayReadV3/Enter:0N
 q_next/rnn/lstm_cell/bias/read:0*q_next/rnn/while/lstm_cell/BiasAdd/Enter:0
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc:0Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_1_grad/Mul/Enter:0
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc:0Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/Mul_1/Enter:0
Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc:0Bq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul/Enter:07
q_next/rnn/Minimum:0q_next/rnn/while/Less_1/Enter:0Ќ
Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0Tq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
Fq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc:0Fq_next/gradients/q_next/rnn/while/lstm_cell/concat_grad/ShapeN/Enter:0А
Vq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0Vq_next/gradients/q_next/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0O
"q_next/rnn/lstm_cell/kernel/read:0)q_next/rnn/while/lstm_cell/MatMul/Enter:0=
q_next/rnn/strided_slice_1:0q_next/rnn/while/Less/Enter:0Ј
Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0Rq_next/gradients/q_next/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
Dq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Dq_next/gradients/q_next/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter:0Rq_next/rnn/while/Enter:0Rq_next/rnn/while/Enter_1:0Rq_next/rnn/while/Enter_2:0Rq_next/rnn/while/Enter_3:0Rq_next/rnn/while/Enter_4:0Rq_next/gradients/f_count_1:0Zq_next/rnn/strided_slice_1:0"(
train_op

q_eval/Adam
q_next/Adam"u
regularization_losses\
Z
+q_eval/weights/Regularizer/l2_regularizer:0
+q_next/weights/Regularizer/l2_regularizer:0эЏ_&