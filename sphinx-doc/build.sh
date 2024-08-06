rm -r ../docs/*
make html
echo "reorganizing directories ..."
rm -r ../docs/doctrees
cp -r ../docs/html/* ../docs
rm -r ../docs/html
echo "done."
