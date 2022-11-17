rm -rf metadata.jsonl
touch metadata.jsonl
for x in $(ls *.jpg);do
  echo "{\"file_name\":\"${x}\", \"name\":\"${x}\"}" >> metadata.jsonl
done