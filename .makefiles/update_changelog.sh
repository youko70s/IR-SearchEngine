NOW="$(date +'%B %d, %Y')"
current_tag=$(git tag --sort=-version:refname | sed -n 1p)
previous_tag=$(git tag --sort=-version:refname | sed -n 2p)
# remove the prefix v
echo "## ${current_tag#v} ($NOW)" > tmpfile
git log --pretty=format:"  - %s" "${previous_tag}"..."${current_tag}" >> tmpfile
echo "" >> tmpfile
echo "" >> tmpfile
cat CHANGELOG.md >> tmpfile
mv tmpfile CHANGELOG.md