for file in *.yaml; do
  newfile=$(echo "$file" | sed -E 's/(X[^-]+)-(H[^-.]+)/\2-\1/')
  if [[ "$file" != "$newfile" ]]; then
    mv "$file" "$newfile"
  fi
done
