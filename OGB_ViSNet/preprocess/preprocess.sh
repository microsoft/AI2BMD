ulimit -c unlimited

[ -z "${data_path}" ] && data_path='ogb2022-dataset/Pretrained_3D_ViSNet_dataset'
[ -z "${rdkit_data_path}" ] && rdkit_data_path='ogb2022-dataset/Pretrained_3D_ViSNet_dataset/rdkit_data/'
[ -z "${tc}" ] && tc=True

if [ -f "${data_path}/pcqm4m-v2-train.sdf" ]; then
    echo "Data path ${data_path}/pcqm4m-v2-train.sdf already exists. Skipping downloading."
else
    mkdir -p $data_path && \
    wget -P $data_path http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz && \
    tar -xvf $data_path/pcqm4m-v2-train.sdf.tar.gz -C $data_path && \
    rm -r $data_path/pcqm4m-v2-train.sdf.tar.gz
fi

if [ "$tc" = True ]; then
    python OGB_ViSNet/preprocess/batch_rdkit.py $data_path $rdkit_data_path $tc 0 147432
else
    # ! We recommend to use distributed machines to preprocess the dataset.
    # for i in 0 100
    # do
    # python OGB_ViSNet/preprocess/batch_rdkit.py $data_path $rdkit_data_path $tc $i 38000
    # done
    python OGB_ViSNet/preprocess/batch_rdkit.py $data_path $rdkit_data_path $tc 0 3746620
fi

python OGB_ViSNet/preprocess/combine.py $rdkit_data_path $tc

mkdir -p $data_path/raw

cp -r $rdkit_data_path/*.pkl $data_path/raw

cp $data_path/pcqm4m-v2/split_dict.pt $data_path