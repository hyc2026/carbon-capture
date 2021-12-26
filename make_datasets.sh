CYCLE_NUM=10
NUM_EACH_CYCLE=10

for((i=1;i<=$CYCLE_NUM;i++));
do
python make_datasets.py $i $NUM_EACH_CYCLE;
done