PREFIX="n02108915-French_bulldog"
for((i=0;i<"$1";i++)); do
  echo "Building dir ${i}"
  rm -rf "${PREFIX}/${i}"
  mkdir "${PREFIX}/${i}"
done

while IFS=' ' read -ra line; do
  cp "${PREFIX}/${line[0]}" "${PREFIX}/${line[1]}/"
done < ./result.txt

rm -rf "${PREFIX}/poor_image"
mkdir "${PREFIX}/poor_image"
while read -r line; do
  cp "${PREFIX}/${line}" "${PREFIX}/poor_image/"
done < ./poor_image.txt

rm -rf "${PREFIX}/cluster"
mkdir "${PREFIX}/cluster"
while read -r line; do
  cp "${PREFIX}/${line}" "${PREFIX}/cluster/"
done < ./cluster.txt