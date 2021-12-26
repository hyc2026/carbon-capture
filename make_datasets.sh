CYCLE_NUM=20
NUM_EACH_CYCLE=100

for((i=1;i<=$CYCLE_NUM;i++));
do
    {
    python make_datasets.py $i $NUM_EACH_CYCLE;
    } &
done