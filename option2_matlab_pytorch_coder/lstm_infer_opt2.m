% lstm_infer_opt2.m  — codegen entry point for Option 2
% Input : single [1 x 75 x 3]   (batch x seq x features)
% Output: int32  [1 x 75]         (per-step class index)
function out = lstm_infer_opt2(in1)
%#codegen
persistent ptNet;
if isempty(ptNet)
    ptNet = loadPyTorchExportedProgram('/Users/arkadiyturevskiy/Documents/Claude/Codegen/LSTMSeqToSeqModel.pt2');
end
out = ptNet.invoke(in1);
end
