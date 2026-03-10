%% generate_code_opt4.m
%  Option 4: Export to ONNX -> importNetworkFromONNX -> dlnetwork -> MATLAB Coder
%  Generates library-free C code using Embedded Coder with DeepLearningConfig('none')
%
%  Workflow:
%    LSTMSeqToSeqModel.onnx -> importNetworkFromONNX -> dlnetwork -> codegen -> C
%
%  Usage:
%    cd option4_onnx_matlab
%    run('generate_code_opt4.m')

PROJECT  = fileparts(fileparts(mfilename('fullpath')));
OPT4_DIR = fullfile(PROJECT, 'option4_onnx_matlab');
ONNX_FILE = fullfile(OPT4_DIR, 'LSTMSeqToSeqModel.onnx');
PT2_FILE  = fullfile(PROJECT, 'LSTMSeqToSeqModel.pt2');
OUT_DIR  = fullfile(OPT4_DIR, 'codegen_opt4');

fprintf('=== Option 4: ONNX -> importNetworkFromONNX -> dlnetwork -> MATLAB Coder ===\n');
fprintf('  ONNX   : %s\n', ONNX_FILE);
fprintf('  Output : %s\n', OUT_DIR);

%% Step 1: Import model from ONNX
fprintf('\n  Importing ONNX model...\n');
onnxNet = importNetworkFromONNX(ONNX_FILE);
fprintf('  SUCCESS: %d layers\n', length(onnxNet.Layers));

% Show ONNX network learnables
fprintf('  ONNX network learnables:\n');
onnxLearnables = onnxNet.Learnables;
for i = 1:height(onnxLearnables)
    fprintf('    %s / %s : [%s]\n', onnxLearnables.Layer{i}, ...
        onnxLearnables.Parameter{i}, num2str(size(extractdata(onnxLearnables.Value{i}))));
end

%% Step 2: Build clean codegen-compatible dlnetwork using ONNX-derived weights
fprintf('\n  Building clean dlnetwork from ONNX weights...\n');

% Extract LSTM params from ONNX network
onnxIW = [];
onnxRW = [];
onnxB  = [];
fcW    = [];
fcB    = [];

for i = 1:height(onnxLearnables)
    lname = onnxLearnables.Layer{i};
    pname = onnxLearnables.Parameter{i};
    val   = extractdata(onnxLearnables.Value{i});
    if (contains(lname,'lstm') || contains(lname,'LSTM')) && strcmp(pname,'InputWeights')
        onnxIW = val;
    elseif (contains(lname,'lstm') || contains(lname,'LSTM')) && strcmp(pname,'RecurrentWeights')
        onnxRW = val;
    elseif (contains(lname,'lstm') || contains(lname,'LSTM')) && strcmp(pname,'Bias')
        onnxB  = val;
    elseif contains(pname,'MatMul')
        fcW = val;
    end
end

% FC bias from PT2 reference
if isempty(fcB)
    fprintf('  FC bias not found in ONNX loading from .pt2 reference\n');
    netRef = importNetworkFromPyTorch(PT2_FILE);
    for i = 1:length(netRef.Layers)
        L = netRef.Layers(i);
        if isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
            fcB = L.Bias;
            break;
        end
    end
end

fprintf('  LSTM InputWeights:     [%s]\n', num2str(size(onnxIW)));
fprintf('  LSTM RecurrentWeights: [%s]\n', num2str(size(onnxRW)));
fprintf('  LSTM Bias:             [%s]\n', num2str(size(onnxB)));
fprintf('  FC Weights:            [%s]\n', num2str(size(fcW)));
fprintf('  FC Bias:               [%s]\n', num2str(size(fcB)));

%% Step 3: Build fresh dlnetwork with weights set via layer constructors
newInputLayer = sequenceInputLayer(3, 'Name', 'input');
newLSTMLayer  = lstmLayer(50, 'OutputMode', 'sequence', 'Name', 'lstm', ...
    'InputWeights',     single(onnxIW), ...
    'RecurrentWeights', single(onnxRW), ...
    'Bias',             single(onnxB));
newFCLayer    = fullyConnectedLayer(5, 'Name', 'fc', ...
    'Weights', single(fcW), ...
    'Bias',    single(fcB));

lg = layerGraph([newInputLayer; newLSTMLayer; newFCLayer]);
netCodegen = dlnetwork(lg);
fprintf('  Weights set via layer constructors\n');
%% Step 5: Validate
fprintf('\n  Validating network...\n');
X_test = dlarray(single(rand(3,75,1)), 'CTB');
Y_test = predict(netCodegen, X_test);
fprintf('  Output size: [%s]\n', num2str(size(Y_test)));

%% Step 6: Save network
netFile = fullfile(OPT4_DIR, 'lstm_net4.mat');
save(netFile, 'netCodegen');
fprintf('  Saved network: %s\n', netFile);

%% Step 7: Write entry-point function
epFile = fullfile(OPT4_DIR, 'lstm_infer_opt4.m');
fid = fopen(epFile, 'w');
fprintf(fid, 'function pred = lstm_infer_opt4(X)\n');
fprintf(fid, '%%#codegen\n');
fprintf(fid, 'persistent net;\n');
fprintf(fid, 'if isempty(net)\n');
fprintf(fid, '    net = coder.loadDeepLearningNetwork(''lstm_net4.mat'', ''net4'');\n');
fprintf(fid, 'end\n');
fprintf(fid, 'Xdl = dlarray(X, ''CT'');\n');
fprintf(fid, 'Y = predict(net, Xdl);\n');
fprintf(fid, 'Ydata = extractdata(Y);\n');
fprintf(fid, '[~, idx] = max(Ydata, [], 1);\n');
fprintf(fid, 'pred = int32(idx) - int32(1);\n');
fprintf(fid, 'end\n');
fclose(fid);
fprintf('  Entry-point: %s\n', epFile);

%% Step 8: Configure MATLAB Coder
cfg = coder.config('lib');
cfg.TargetLang         = 'C';
cfg.GenerateReport     = true;
cfg.EnableOpenMP       = false;
cfg.DeepLearningConfig = coder.DeepLearningConfig('none');

%% Step 9: Input type
inType = coder.typeof(single(zeros(3,75)), [3 75], [false false]);

%% Step 10: Run codegen
fprintf('\n  Running codegen (dlnetwork ONNX-origin -> C)...\n');
[~,~] = mkdir(OUT_DIR);
prevDir = cd(OPT4_DIR);
cleanup = onCleanup(@() cd(prevDir));

try
    codegen('-config', cfg, '-d', OUT_DIR, epFile, '-args', {inType});
    fprintf('  SUCCESS\n');
catch ME
    cFiles = dir(fullfile(OUT_DIR, '*.c'));
    if ~isempty(cFiles)
        fprintf('  NOTE: Build step failed (%s)\n', ME.message);
        fprintf('  C source files generated (%d files)\n', length(cFiles));
    else
        fprintf('  FAILED: %s\n', ME.message);
        exit(1);
    end
end

fprintf('\n  Generated files:\n');
d = [dir(fullfile(OUT_DIR,'*.c')); dir(fullfile(OUT_DIR,'*.h'))];
for i = 1:length(d)
    fprintf('    %s  (%d bytes)\n', d(i).name, d(i).bytes);
end

fprintf('\n=== Option 4 codegen complete ===\n');
exit(0);
