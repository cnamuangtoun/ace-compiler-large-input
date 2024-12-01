#!/usr/bin/python3

import os
import sys
import argparse
import subprocess
import datetime
import signal

def write_log(info, log):
    print(info[:-1])
    log.write(info)
    log.flush()
    return

def time_and_memory(outputs):
    result = outputs.strip('"').split(' ')
    return result[0], result[1]

def ace_compile_and_run_onnx(cwd, cmplr_path, onnx_path, onnx_model, log, debug):
    script_dir = os.path.dirname(__file__)
    onnx2c = os.path.join(script_dir, 'onnx2c.py')
    if not os.path.exists(onnx2c):
        print(onnx2c, 'does not exist')
        sys.exit(-1)
    model_file = os.path.join(onnx_path, onnx_model)
    if not os.path.exists(model_file):
        print(model_file, 'does not exist')
        return
    info = onnx_model + ':\n'
    write_log(info, log)
    model_base = onnx_model.split('.')[0]
    main_c = os.path.join(cwd, model_base + '.main.c')
    exec_file = os.path.join(cwd, model_base + '.ace')
    onnx_c = os.path.join(cwd, onnx_model + '.c')
    wfile = model_base + '.weight'
    
    # Compile ONNX
    cmds = ['python3', onnx2c, '-mp', model_file, '-op', main_c]
    if debug:
        print(' '.join(cmds))
    ret = subprocess.run(cmds, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        info += 'Failed to generate main.c\n'
        if debug:
            info = ' '.join(cmds) + '\n'
        write_log(info, log)
        return
    
    # ACE compile
    os.environ["RTLIB_BTS_EVEN_POLY"] = "1"
    os.environ["RTLIB_TIMING_OUTPUT"] = "stdout"
    cmds = ['time', '-f', '\"%e %M\"', os.path.join(cmplr_path, 'bin', 'fhe_cmplr')]
    cmds.extend([model_file, '-P2C:fp:df=' + wfile + ':lib=ant'])
    cmds.extend(['-SIHE:relu_vr_def=21:relu_mul_depth=13', '-CKKS:sk_hw=192', '-o', onnx_c])
    cmds.extend(['-O2A:ts', '-FHE_SCHEME:ts', '-VEC:ts:rtt:conv_fast:toeplitz:td=3:tia:tib', '-SIHE:ts:rtt'])
    cmds.extend(['-CKKS:ts:q0=60:sf=56', '-POLY:ts:rtt', '-P2C:ts'])
    if debug:
        print(' '.join(cmds))
    if os.path.exists(wfile):
        os.remove(wfile)
    ret = subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode == 0:
        time, memory = time_and_memory(ret.stderr.decode().splitlines()[0])
        info = ' '*(len(onnx_model)+2) + 'ACE: Time = ' + str(time) \
            + '(s) Memory = ' + str("%.2f" % (int(memory)/1000000)) + '(Gb)\n'
        write_log(info, log)
        # Handle execution
        cmds = ['time', '-f', '\"%e %M\"', 'cc', main_c, onnx_c]
        cmds.extend(['-I', os.path.join(cmplr_path, 'rtlib/include')])
        cmds.extend(['-I', os.path.join(cmplr_path, 'rtlib/include/rt_ant')])
        cmds.append(os.path.join(cmplr_path, 'rtlib/lib/libFHErt_ant.a'))
        cmds.append(os.path.join(cmplr_path, 'rtlib/lib/libFHErt_common.a'))
        cmds.extend(['-lgmp', '-lm', '-o', exec_file])
        if debug:
            print(' '.join(cmds))
        ret = subprocess.run(cmds, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if ret.returncode == 0:
            # Execution successful, run the executable
            cmds = ['time', '-f', '\"%e %M\"', exec_file]
            if debug:
                print(' '.join(cmds))
            ret = subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = ret.stdout.decode().splitlines()
            if ret.returncode == 0:
                time, memory = time_and_memory(ret.stderr.decode().splitlines()[0])
                info = 'Exec: Time = ' + str(time) + '(s) TotalMemory = ' \
                    + str("%.1f" % (int(memory)/1000000)) + '(Gb)\n'
                write_log(info, log)
            else:
                info = 'Exec: failed\n'
                if ret.returncode > 128:
                    info += ' due to ' + signal.Signals(ret.returncode - 128).name
                write_log(info, log)
        else:
            info = 'GCC: failed\n'
            write_log(info, log)
    else:
        info = 'ACE: failed\n'
        write_log(info, log)

    return

def test_perf_ace(cwd, cmplr_path, onnx_path, log, debug):
    info = '-------- ACE --------\n'
    write_log(info, log)
    os.chdir(cwd)
    model_files = [f for f in os.listdir(onnx_path) if os.path.isfile(os.path.join(onnx_path, f))]
    model_files.sort()
    for onnx_model in model_files:
        ace_compile_and_run_onnx(cwd, cmplr_path, onnx_path, onnx_model, log, debug)
    return

def main():
    parser = argparse.ArgumentParser(description='Run performance data for ACE Framework')
    parser.add_argument('-c', '--cmplr', metavar='PATH', help='Path to the ACE compiler')
    parser.add_argument('-m', '--model', metavar='PATH', help='Path to the ONNX models')
    parser.add_argument('-f', '--file', metavar='PATH', help='Run single ONNX model only')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='Print out debug info')
    args = parser.parse_args()
    debug = args.debug
    
    # ACE compiler path
    cmplr_path = '/app/ace_cmplr'
    if args.cmplr is not None:
        cmplr_path = os.path.abspath(args.cmplr)
    if not os.path.exists(cmplr_path):
        print(cmplr_path, 'does not exist! Please provide the correct ACE compiler path!')
        sys.exit(-1)
    
    # ONNX model path
    onnx_path = '/app/model'
    if args.model is not None:
        onnx_path = os.path.abspath(args.model)
    if not os.path.exists(onnx_path):
        print(onnx_path, 'does not exist! Pre-trained ONNX model files are missing!')
        sys.exit(-1)
    
    cwd = os.getcwd()
    date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_file_name = date_time + '.log'
    log = open(os.path.join(cwd, log_file_name), 'w')
    
    # Run tests
    if args.file is not None:
        info = '-------- ACE --------\n'
        write_log(info, log)
        test = os.path.basename(args.file)
        onnx_path = os.path.abspath(os.path.dirname(args.file))
        ace_compile_and_run_onnx(cwd, cmplr_path, onnx_path, test, log, debug)
    else:
        test_perf_ace(cwd, cmplr_path, onnx_path, log, debug)
    
    info = '-------- Done --------\n'
    write_log(info, log)
    log.close()
    return

if __name__ == "__main__":
    main()