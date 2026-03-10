function pred = lstm_infer_opt4(X)
%#codegen
persistent net;
if isempty(net)
    net = coder.loadDeepLearningNetwork('lstm_net4.mat', 'net4');
end
Xdl = dlarray(X, 'CT');
Y = predict(net, Xdl);
Ydata = extractdata(Y);
[~, idx] = max(Ydata, [], 1);
pred = int32(idx) - int32(1);
end
