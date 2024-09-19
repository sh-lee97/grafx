make html
touch build/html/.nojekyll
rsync -abviuzP build/html/* ../docs
