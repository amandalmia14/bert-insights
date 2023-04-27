
for FILE in $(find sh-scripts/ -name '*.sh');
do
    if [[ ${FILE} != *"reproduce-glue"* ]]
    then
        echo ${FILE}
        chmod +x ${FILE}
        sbatch ${FILE}
        sleep 15 # pause to be kind to the scheduler
    fi
done