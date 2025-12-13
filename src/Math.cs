using System;

/// ===================================================================
/// 数学演算：CPU実装のニューラルネットワーク基本関数
/// 
/// 機能：
/// - 行列乗算（matmul）
/// - 活性化関数（ReLU）
/// - ソフトマックス
/// - クロスエントロピー損失
/// ===================================================================
public static class MathOps
{
    /// <summary>
    /// 行列乗算：C = A @ B
    /// 
    /// 3重ループによるO(n³)実装
    /// A: (m, k)
    /// B: (k, n)
    /// C: (m, n)
    /// </summary>
    public static void Matmul(Tensor C, Tensor A, Tensor B)
    {
        int m = A.Shape[0];
        int k = A.Shape[1];
        int n = B.Shape[1];
        
        C.Zero();
        
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int p = 0; p < k; p++)
                {
                    sum += A.Get(i, p) * B.Get(p, j);
                }
                C.Set(i, j, sum);
            }
        }
    }
    
    /// <summary>
    /// ReLU活性化関数
    /// 
    /// ReLU(x) = max(0, x)
    /// ニューラルネットワークで最も一般的な活性化関数
    /// </summary>
    public static void Relu(Tensor x)
    {
        for (int i = 0; i < x.Size; i++)
        {
            if (x.Get(i) < 0f)
            {
                x.Set(i, 0f);
            }
        }
    }
    
    /// <summary>
    /// ReLU逆伝播
    /// 
    /// dx = dy * (x > 0 ? 1 : 0)
    /// </summary>
    public static void ReluBackward(Tensor dx, Tensor dy, Tensor x)
    {
        for (int i = 0; i < x.Size; i++)
        {
            dx.Set(i, x.Get(i) > 0f ? dy.Get(i) : 0f);
        }
    }
    
    /// <summary>
    /// ソフトマックス関数
    /// 
    /// 各行ごとに処理：
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j))
    /// 
    /// 数値安定性のため最大値を引いてからexp計算
    /// </summary>
    public static void Softmax(Tensor x)
    {
        int batchSize = x.Shape[0];
        int vocabSize = x.Shape[1];
        
        for (int b = 0; b < batchSize; b++)
        {
            // ステップ1：行の最大値を見つける
            float maxVal = float.MinValue;
            for (int i = 0; i < vocabSize; i++)
            {
                float val = x.Get(b, i);
                if (val > maxVal)
                    maxVal = val;
            }
            
            // ステップ2：exp計算と合計を計算
            float sum = 0f;
            for (int i = 0; i < vocabSize; i++)
            {
                float val = x.Get(b, i);
                float expVal = (float)Math.Exp(val - maxVal);
                x.Set(b, i, expVal);
                sum += expVal;
            }
            
            // ステップ3：合計で正規化
            for (int i = 0; i < vocabSize; i++)
            {
                float val = x.Get(b, i);
                x.Set(b, i, val / sum);
            }
        }
    }
    
    /// <summary>
    /// クロスエントロピー損失
    /// 
    /// LLM学習の核心：
    /// loss = -sum(target * log(pred))
    /// 
    /// 正解トークン（target）の確率が高いほど、損失が小さい
    /// </summary>
    public static float CrossEntropyLoss(Tensor predictions, Tensor targets)
    {
        float loss = 0f;
        int batchSize = predictions.Shape[0];
        int vocabSize = predictions.Shape[1];
        
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < vocabSize; i++)
            {
                float pred = predictions.Get(b, i);
                float target = targets.Get(b, i);
                
                // target=1, pred=高確率 → loss小
                // target=1, pred=低確率 → loss大
                if (target > 0.5f)
                {
                    loss -= (float)Math.Log(Math.Max(pred, 1e-7f));
                }
            }
        }
        
        return loss / batchSize;
    }
    
    /// <summary>
    /// クロスエントロピー逆伝播
    /// 
    /// dL/dz = pred - target
    /// </summary>
    public static void CrossEntropyBackward(Tensor dz, Tensor predictions, Tensor targets)
    {
        int batchSize = predictions.Shape[0];
        int vocabSize = predictions.Shape[1];
        
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < vocabSize; i++)
            {
                float pred = predictions.Get(b, i);
                float target = targets.Get(b, i);
                dz.Set(b, i, (pred - target) / batchSize);
            }
        }
    }
}
