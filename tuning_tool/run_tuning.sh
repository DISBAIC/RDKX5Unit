#!/bash
CUR_TEST_SHELL=$(readlink -f $0)
COMMON_DIR=$(pwd)
case_list=()
run_case=
first_file=

# tuning tool case
VPM_JSON_NAME=vpm_x5_config.json
CAM_JSON_NAME=cam_x5_config.json
TUNING_CFG_PATH=${COMMON_DIR}/tuning_cfg

function print_usage() {
	echo "run_tuning.sh --list: list all case"
	echo "run_tuning.sh --run [case_index]: run this case"
	echo "run_tuning.sh --vpm [case_index]: edit this case's vpm json"
	echo "run_tuning.sh --cam [case_index]: edit this case's cam json"
	echo "run_tuning.sh --tune 0/1: close/open tuning_server"
	echo "run with [-w 2]: dump 20 yuv from the start"
	echo "run with [-r 1]: send raw to hbplayer"
	exit 1
}

function get_case_list() {
	local _case_list=$(grep -n 'test "$run_case"' $CUR_TEST_SHELL | grep -v "print $4"| awk -F '"' '{print $4}')
	local i=0

	for case in ${_case_list}
	do
		case_list[$i]=${case}
		let "i++"
	done
}

function get_case() {
	if [ -z "$1" ]; then
		echo "Please input the index of the case which you want to run."
		echo "eg: --run 1"
		exit -1
	fi

	if [ "$1" -ge ${#case_list[@]} ]; then
		echo "index $1 not support"
		exit -1
	fi

	run_case=${case_list[$1]}
}

function suit_case_run() {
	get_case "$@"
	shift 1

	local vpm_json_path=${TUNING_CFG_PATH}/${run_case}/${VPM_JSON_NAME}
	local cam_json_path=${TUNING_CFG_PATH}/${run_case}/${CAM_JSON_NAME}

	if test "$run_case" == "imx219_rx2"; then
		echo "Run $run_case"
		${COMMON_DIR}/isp_tuning -v "${vpm_json_path}" -c "${cam_json_path}" "$@"
	
	elif test "$run_case" == "ov5647_rx2"; then
		echo "Run $run_case"
		${COMMON_DIR}/isp_tuning -v "${vpm_json_path}" -c "${cam_json_path}" "$@"

	elif test "$run_case" == "gc4663_rx2"; then
		echo "Run $run_case"
		${COMMON_DIR}/isp_tuning -v "${vpm_json_path}" -c "${cam_json_path}" "$@"

	elif test "$run_case" == "imx477_rx2"; then
		echo "Run $run_case"
		${COMMON_DIR}/isp_tuning -v "${vpm_json_path}" -c "${cam_json_path}" "$@"

	elif test "$run_case" == "feedback_case"; then
		echo "Run $run_case"
		echo "Notice: this script just support feedback 1080p raw now!!"
		${COMMON_DIR}/isp_tuning -v "${vpm_json_path}" -c "${cam_json_path}" -w 1 "$@"

	fi
}

function open_case_vpm_json() {
	get_case "$@"
	local vpm_json_path=${TUNING_CFG_PATH}/${run_case}/${VPM_JSON_NAME}

	if [ -f ${vpm_json_path} ]; then
		vi ${vpm_json_path}
	else
		echo "file ${vpm_json_path} not exist"
	fi
}

function open_case_cam_json() {
	get_case "$@"
	local cam_json_path=${TUNING_CFG_PATH}/${run_case}/${CAM_JSON_NAME}

	if [ -f ${cam_json_path} ]; then
		vi ${cam_json_path}
	else
		echo "file ${cam_json_path} not exist"
	fi
}

function vtuner_control() {
	if [ -z "$1" ]; then
		echo "Please input --tune 1 to open the vtuner_server."
		exit -1
	fi

	if [ "$1" == "1" ]; then
		echo 1 > /sys/kernel/debug/isp/tune
		echo "Open vtuner_server Success! Please connect VtunerClient"

	elif [ "$1" == "0" ]; then
		echo 0 > /sys/kernel/debug/isp/tune
		echo "Close vtuner_server done!"

	else
		echo "Please input --tune 0/1 to close/open the vtuner_server."
		exit -1
	fi
}

function specify_command() {
	local _command=$1
	shift 1

	if [ "$_command" == "--list" ]; then
		echo "all support cases :"
		for (( j=0;j<${#case_list[@]};j+=1 )); do
			local _case=${case_list[$j]}
			echo "$j. ${_case}"
		done

	elif [ "$_command" == "--run" ]; then
		suit_case_run "$@"

	elif [ "$_command" == "--vpm" ]; then
		open_case_vpm_json "$@"

	elif [ "$_command" == "--cam" ]; then
		open_case_cam_json "$@"

	elif [ "$_command" == "--help" ]; then
		print_usage

	elif [ "$_command" == "--tune" ]; then
		vtuner_control "$@"

	else
		echo "invaild cmd input : $_command "
		print_usage

	fi
}

get_case_list
specify_command "$@"
