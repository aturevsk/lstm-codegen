%% generate_code_opt2.m
%  MATLAB Coder Support Package for PyTorch (R2026a)
%  Generates library-free C code from LSTMSeqToSeqModel.pt2
%
%  Workflow:
%    loadPyTorchExportedProgram(.pt2) -> MLIR lowering -> MATLAB Coder -> C
%
%  Usage:
%    cd option2_matlab_pytorch_coder
%    run('generate_code_opt2.m')

PROJECT  = fileparts(fileparts(mfilename('fullpath')));
OPT2_DIR = fullfile(PROJECT, 'option2_matlab_pytorch_coder');
PT2_FILE = fullfile(PROJECT, 'LSTMSeqToSeqModel.pt2');
OUT_DIR  = fullfile(OPT2_DIR, 'codegen_opt2');

fprintf('=== Option 2: MATLAB Coder Support Package for PyTorch ===\n');
fprintf('  Model  : %s\n', PT2_FILE);
fprintf('  Output : %s\n', OUT_DIR);

%% Step 1: Entry-point function
% This function is the codegen entry point per PyTorchExportedProgram docs.
% MATLAB Coder's matlabCodegenRedirect hooks in MLIR lowering automatically.
epFile = fullfile(OPT2_DIR, 'lstm_infer_opt2.m');
fid = fopen(epFile, 'w');
fprintf(fid, '%% lstm_infer_opt2.m  — codegen entry point for Option 2\n');
fprintf(fid, '%% Input : single [1 x 75 x 3]   (batch x seq x features)\n');
fprintf(fid, '%% Output: int32  [1 x 75]         (per-step class index)\n');
fprintf(fid, 'function out = lstm_infer_opt2(in1)\n');
fprintf(fid, '%%#codegen\n');
fprintf(fid, 'persistent ptNet;\n');
fprintf(fid, 'if isempty(ptNet)\n');
fprintf(fid, '    ptNet = loadPyTorchExportedProgram(''%s'');\n', PT2_FILE);
fprintf(fid, 'end\n');
fprintf(fid, 'out = ptNet.invoke(in1);\n');
fprintf(fid, 'end\n');
fclose(fid);
fprintf('  Entry-point: %s\n', epFile);

%% Step 2: Configure MATLAB Coder
cfg = coder.config('lib');
cfg.TargetLang         = 'C';
cfg.GenerateReport     = true;
cfg.EnableOpenMP       = false;   % library-free: no OpenMP threads

%% Step 3: Define input type  [1 × 75 × 3] single, fixed size
inType = coder.typeof(single(zeros(1,75,3)), [1 75 3], [false false false]);

%% Step 4: Run codegen
fprintf('\n  Running codegen (MLIR -> C)...\n');
[~,~] = mkdir(OUT_DIR);

try
    codegen('-config', cfg, ...
            '-d', OUT_DIR, ...
            '-I', OPT2_DIR, ...
            epFile, ...
            '-args', {inType});
    fprintf('  SUCCESS: C code generated in %s\n', OUT_DIR);
catch ME
    % C files are generated even if the final build step fails
    cFiles = dir(fullfile(OUT_DIR, '*.c'));
    if ~isempty(cFiles)
        fprintf('  NOTE: Build step failed (%s)\n', ME.message);
        fprintf('  C source files were generated (%d files):\n', length(cFiles));
        for i = 1:length(cFiles)
            fprintf('    %s  (%d bytes)\n', cFiles(i).name, cFiles(i).bytes);
        end
    else
        fprintf('  FAILED: %s\n', ME.message);
        exit(1);
    end
end

%% Step 5: List generated files
fprintf('\n  Generated files:\n');
d = dir(fullfile(OUT_DIR, '*.c'));
for i = 1:length(d)
    fprintf('    %s  (%d bytes)\n', d(i).name, d(i).bytes);
end
d = dir(fullfile(OUT_DIR, '*.h'));
for i = 1:length(d)
    fprintf('    %s\n', d(i).name);
end

fprintf('\n=== Option 2 codegen complete ===\n');
exit(0);
