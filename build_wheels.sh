sudo rm -rf build
sudo rm -rf tempest.egg-info/
sudo rm -rf dist
sudo rm -rf wheelhouse

docker build -t tempest-12-6 -f build_scripts/build_12_6/Dockerfile .
docker run --gpus all -v $(pwd):/project tempest-12-6
