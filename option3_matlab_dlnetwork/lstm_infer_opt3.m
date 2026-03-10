% lstm_infer_opt3.m  — codegen entry point for Option 3
% importNetworkFromPyTorch -> dlnetwork -> MATLAB Coder
%
% Input : single [3 x 75]   (Channels x Time, one sample)
% Output: int32  [1 x 75]   (per-step class index, 0-indexed)
function pred = lstm_infer_opt3(X)
%#codegen
persistent net;
if isempty(net)
    net = coder.loadDeepLearningNetwork('lstm_net3.mat', 'net3');
end
% dlarray format: CT (Channels x Time), batch-less for codegen
Xdl = dlarray(X, 'CT');
Y = predict(net, Xdl);      % [5 x 75] float32 logits
Ydata = extractdata(Y);     % [5 x 75] float32
% Argmax along dim 1 (class axis) -> 1-indexed, convert to 0-indexed
[~, idx] = max(Ydata, [], 1);
pred = int32(idx) - int32(1);
end
