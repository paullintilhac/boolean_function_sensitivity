while getopts d:f:l:h: arg
do
    case "${arg}" in
        d) hidden_dim=${OPTARG};;
        f) ff_dim=${OPTARG};;
        l) layers=${OPTARG};;
        h) heads=${OPTARG};;
    esac
done

echo "hidden_dim: $hidden_dim";
echo "ff_dim: $ff_dim";
echo "layers: $layers";
echo "heads: $heads";

echo after printing args
echo "d_$hidden_dim-f_$ff_dim-l_$layers-h_$heads"
fn="d_$hidden_dim-f_$ff_dim-l_$layers-h_$heads"
if [ ! -d $fn ] 
then
echo making new dir with these hyperparams
mkdir -p $fn
fi
cd $fn
echo should be in folder now
ls
pwd
sbatch ../script.sh -d $hidden_dim -f $ff_dim -l $layers -h $heads
cd ..
