#!/bin/bash

# Array of input files
input_dirs=(
  "../runs/atax/M1/N256/U/N16"
  "../runs/atax/M1/N256/M/N16"
  "../runs/gemm/M256/N1/U/N16"
  "../runs/gemm/M256/N1/M/N16"
)

# Scripts
roi_py="../../../../../deps/snitch_cluster/util/bench/roi.py"
visualize_py="../../../../../deps/snitch_cluster/util/bench/visualize.py"

# Process each file
for input_dir in "${input_dirs[@]}"; do
  # Construct the output file path by replacing the input directory with the output directory
  output_dir="./${input_dir#../runs/}"

  mkdir -p $output_dir

  app=$(echo "${input_dir##*/runs/}" | cut -d'/' -f1)

  # Determine if the run is unicast or multicast
  if [[ "$input_dir" == */U/* ]]; then
    roi="roi/U-$app.json"
  elif [[ "$input_dir" == */M/* ]]; then
    roi="roi/M-$app.json"
  else
    echo "Unknown mode in $input_dir. Skipping..."
    continue
  fi

  # Filter only DMA traces
  python $roi_py $input_dir/logs/perf.json $roi --cfg ../../../cfg/M-Q8C4.hjson -o $output_dir/roi.json
  python $visualize_py $output_dir/roi.json -o $output_dir/trace.json

  # Remove zero event from trace
  jq '.traceEvents |= map(select(.name != "zero"))' $output_dir/trace.json > $output_dir/tmp.json

  # Subtract the minimum start time from all events, to base the trace on zero time
  jq '
    ( .traceEvents | map(select(has("ts")) | .ts) | min ) as $minTs |
    .traceEvents |= map(
      if has("ts") then .ts |= ((. - $minTs) * 1000 | round) / 1000 else . end
    )
  ' "$output_dir/tmp.json" > "$output_dir/trace.json"

  echo "Processed: $input_dir -> $output_dir"
done

# Merge files
jq -s '
  # Combine traceEvents from both files
  {
    traceEvents: (
      .[0].traceEvents +
      (.[1].traceEvents | map(.pid = 1))
    )
  }
' atax/M1/N256/U/N16/trace.json atax/M1/N256/M/N16/trace.json > atax/M1/N256/N16_trace.json

jq -s '
  # Combine traceEvents from both files
  {
    traceEvents: (
      .[0].traceEvents +
      (.[1].traceEvents | map(.pid = 1))
    )
  }
' gemm/M256/N1/U/N16/trace.json gemm/M256/N1/M/N16/trace.json > gemm/M256/N1/N16_trace.json

jq -s '
  # Combine traceEvents from both files
  {
    traceEvents: (
      .[0].traceEvents +
      (.[1].traceEvents | map(.pid = 1)) +
      (.[2].traceEvents | map(.pid = 2)) +
      (.[3].traceEvents | map(.pid = 3))
    )
  }
' atax/M1/N256/U/N16/trace.json atax/M1/N256/M/N16/trace.json gemm/M256/N1/U/N16/trace.json gemm/M256/N1/M/N16/trace.json > trace.json
