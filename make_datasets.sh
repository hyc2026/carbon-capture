CYCLE_NUM=4
NUM_EACH_CYCLE=50

for((i=1;i<=$CYCLE_NUM;i++));
do
    {
    python make_datasets.py $i $NUM_EACH_CYCLE;
    } &
done
