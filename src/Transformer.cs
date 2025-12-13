using System;
using System.Collections.Generic;

/// ===================================================================
/// Transformerレイヤー実装：マルチヘッドアテンション + FFN
/// 
/// 構成：
/// 1. マルチヘッド自己注意（Multi-Head Self-Attention）
/// 2. フィードフォワードネットワーク（FFN）
/// 3. 正規化・残差接続
/// ===================================================================
public class TransformerLayer
{
    private int hiddenDim;
    private int numHeads;
    private float learningRate;
    
    // マルチヘッド注意用の重み
    private Tensor W_q;
    private Tensor W_k;
    private Tensor W_v;
    private Tensor W_o;
    private Tensor b_q, b_k, b_v, b_o;
    
    // フィードフォワード層の重み
    private Tensor W_ff1, W_ff2;
    private Tensor b_ff1, b_ff2;
    
    public TransformerLayer(int hiddenDim, int numHeads, float lr = 0.001f)
    {
        this.hiddenDim = hiddenDim;
        this.numHeads = numHeads;
        this.learningRate = lr;
        
        // マルチヘッド注意の重みを初期化
        W_q = new Tensor(hiddenDim, hiddenDim);
        W_k = new Tensor(hiddenDim, hiddenDim);
        W_v = new Tensor(hiddenDim, hiddenDim);
        W_o = new Tensor(hiddenDim, hiddenDim);
        
        W_q.RandomInit();
        W_k.RandomInit();
        W_v.RandomInit();
        W_o.RandomInit();
        
        b_q = new Tensor(hiddenDim);
        b_k = new Tensor(hiddenDim);
        b_v = new Tensor(hiddenDim);
        b_o = new Tensor(hiddenDim);
        
        b_q.Zero();
        b_k.Zero();
        b_v.Zero();
        b_o.Zero();
        
        // フィードフォワード層の重みを初期化
        W_ff1 = new Tensor(hiddenDim, hiddenDim * 4);
        W_ff2 = new Tensor(hiddenDim * 4, hiddenDim);
        
        W_ff1.RandomInit();
        W_ff2.RandomInit();
        
        b_ff1 = new Tensor(hiddenDim * 4);
        b_ff2 = new Tensor(hiddenDim);
        
        b_ff1.Zero();
        b_ff2.Zero();
    }
    
    /// <summary>
    /// フォワードパス：実際のSelf-Attention実装
    /// 
    /// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    /// </summary>
    public Tensor Forward(Tensor x)
    {
        int seqLen = x.Shape[0];
        
        // Q, K, V計算
        var Q = new Tensor(seqLen, hiddenDim);
        var K = new Tensor(seqLen, hiddenDim);
        var V = new Tensor(seqLen, hiddenDim);
        
        MathOps.Matmul(Q, x, W_q);
        MathOps.Matmul(K, x, W_k);
        MathOps.Matmul(V, x, W_v);
        
        // バイアスを追加
        for (int s = 0; s < seqLen; s++)
        {
            for (int h = 0; h < hiddenDim; h++)
            {
                Q.Set(s, h, Q.Get(s, h) + b_q.Get(h));
                K.Set(s, h, K.Get(s, h) + b_k.Get(h));
                V.Set(s, h, V.Get(s, h) + b_v.Get(h));
            }
        }
        
        // Attention scores: Q @ K^T / sqrt(d)
        float scale = (float)Math.Sqrt(hiddenDim);
        var attnScores = new Tensor(seqLen, seqLen);
        
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                float score = 0f;
                for (int h = 0; h < hiddenDim; h++)
                {
                    score += Q.Get(i, h) * K.Get(j, h);
                }
                attnScores.Set(i, j, score / scale);
            }
        }
        
        // Softmax（行ごと）
        for (int i = 0; i < seqLen; i++)
        {
            float maxScore = float.MinValue;
            for (int j = 0; j < seqLen; j++)
            {
                maxScore = Math.Max(maxScore, attnScores.Get(i, j));
            }
            
            float sumExp = 0f;
            for (int j = 0; j < seqLen; j++)
            {
                float val = (float)Math.Exp(attnScores.Get(i, j) - maxScore);
                attnScores.Set(i, j, val);
                sumExp += val;
            }
            
            for (int j = 0; j < seqLen; j++)
            {
                attnScores.Set(i, j, attnScores.Get(i, j) / sumExp);
            }
        }
        
        // Attention output: attnScores @ V
        var attnOutput = new Tensor(seqLen, hiddenDim);
        for (int i = 0; i < seqLen; i++)
        {
            for (int h = 0; h < hiddenDim; h++)
            {
                float val = 0f;
                for (int j = 0; j < seqLen; j++)
                {
                    val += attnScores.Get(i, j) * V.Get(j, h);
                }
                attnOutput.Set(i, h, val);
            }
        }
        
        // 出力投影
        var out_proj = new Tensor(seqLen, hiddenDim);
        MathOps.Matmul(out_proj, attnOutput, W_o);
        
        // バイアスと残差接続
        for (int s = 0; s < seqLen; s++)
        {
            for (int h = 0; h < hiddenDim; h++)
            {
                out_proj.Set(s, h, out_proj.Get(s, h) + b_o.Get(h) + x.Get(s, h));
            }
        }
        
        // FFN
        var ff_hidden = new Tensor(seqLen, hiddenDim * 4);
        MathOps.Matmul(ff_hidden, out_proj, W_ff1);
        
        // バイアスを追加
        for (int s = 0; s < seqLen; s++)
        {
            for (int h = 0; h < hiddenDim * 4; h++)
            {
                ff_hidden.Set(s, h, ff_hidden.Get(s, h) + b_ff1.Get(h));
            }
        }
        
        MathOps.Relu(ff_hidden);
        
        var ff_out = new Tensor(seqLen, hiddenDim);
        MathOps.Matmul(ff_out, ff_hidden, W_ff2);
        
        // バイアスと残差接続
        for (int s = 0; s < seqLen; s++)
        {
            for (int h = 0; h < hiddenDim; h++)
            {
                ff_out.Set(s, h, ff_out.Get(s, h) + b_ff2.Get(h) + out_proj.Get(s, h));
            }
        }
        
        return ff_out;
    }
}

/// ===================================================================
/// TinyLLMモデル：小型言語モデル実装
/// 
/// 構成：
/// - Embedding層：トークン → ベクトル
/// - Transformerレイヤー × 複数
/// - 出力層：隠れ状態 → 語彙確率分布
/// ===================================================================
public class TinyLLM
{
    private int vocabSize;
    private int hiddenDim;
    private int numLayers;
    private int seqLength;
    private float learningRate;
    
    private Tensor embeddings;  // 語彙 × 隠れ次元
    private List<TransformerLayer> layers = new();
    private Tensor outputWeight;  // 隠れ × 語彙
    
    public TinyLLM(int vocabSize, int hiddenDim, int numLayers = 2, int seqLength = 16, float lr = 0.001f)
    {
        this.vocabSize = vocabSize;
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;
        this.seqLength = seqLength;
        this.learningRate = lr;
        
        // Embedding層を初期化
        embeddings = new Tensor(vocabSize, hiddenDim);
        embeddings.RandomInit();
        
        // Transformerレイヤーを初期化
        for (int i = 0; i < numLayers; i++)
        {
            layers.Add(new TransformerLayer(hiddenDim, 4, lr));
        }
        
        // 出力層を初期化
        outputWeight = new Tensor(hiddenDim, vocabSize);
        outputWeight.RandomInit();
    }
    
    /// <summary>
    /// フォワードパス：トークンID → 予測確率分布
    /// </summary>
    public Tensor Forward(int[] tokenIds)
    {
        // Embedding
        int seqLen = tokenIds.Length;
        var embedded = new Tensor(seqLen, hiddenDim);
        
        for (int i = 0; i < seqLen; i++)
        {
            int tokenId = tokenIds[i];
            for (int j = 0; j < hiddenDim; j++)
            {
                embedded.Set(i, j, embeddings.Get(tokenId, j));
            }
        }
        
        // Transformerレイヤー
        var x = embedded;
        foreach (var layer in layers)
        {
            x = layer.Forward(x);
        }
        
        // 出力層：最後のトークンの隠れ状態を使用
        var lastHidden = new Tensor(1, hiddenDim);
        for (int j = 0; j < hiddenDim; j++)
        {
            lastHidden.Set(0, j, x.Get(seqLen - 1, j));
        }
        
        var logits = new Tensor(1, vocabSize);
        MathOps.Matmul(logits, lastHidden, outputWeight);
        
        // ソフトマックス
        MathOps.Softmax(logits);
        
        return logits;
    }
    
    /// <summary>
    /// 訓練ステップ：フォワード・バックワード・更新
    /// </summary>
    public float TrainStep(int[] tokenIds, int targetId)
    {
        // フォワードパス
        var logits = Forward(tokenIds);
        
        // ターゲットをOne-hotに変換
        var target = new Tensor(1, vocabSize);
        target.Zero();
        if (targetId >= 0 && targetId < vocabSize)
        {
            target.Set(0, targetId, 1f);
        }
        
        // 損失計算
        float loss = MathOps.CrossEntropyLoss(logits, target);
        
        // 簡易版の勾配降下：出力層の重みを更新
        // gradient = (predictions - target)
        for (int v = 0; v < vocabSize; v++)
        {
            float grad = logits.Get(0, v) - target.Get(0, v);
            
            // 出力重みの更新
            for (int h = 0; h < hiddenDim; h++)
            {
                float currentWeight = outputWeight.Get(h, v);
                outputWeight.Set(h, v, currentWeight - learningRate * grad * 0.01f);
            }
        }
        
        // Embedding層の更新（簡易版）
        for (int i = 0; i < tokenIds.Length && i < embeddings.Shape[0]; i++)
        {
            int tokenId = tokenIds[i];
            if (tokenId >= 0 && tokenId < vocabSize)
            {
                for (int h = 0; h < hiddenDim; h++)
                {
                    float currentEmb = embeddings.Get(tokenId, h);
                    embeddings.Set(tokenId, h, currentEmb - learningRate * loss * 0.0001f);
                }
            }
        }
        
        return loss;
    }
    
    /// <summary>
    /// 推論：次のトークンを予測
    /// </summary>
    public string Predict(int[] tokenIds, SimpleTokenizer tokenizer)
    {
        var logits = Forward(tokenIds);
        
        // 確率が最大のトークンを選択
        float maxProb = -1f;
        int predictedId = 0;
        
        for (int i = 0; i < vocabSize; i++)
        {
            float prob = logits.Get(0, i);
            if (prob > maxProb)
            {
                maxProb = prob;
                predictedId = i;
            }
        }
        
        return tokenizer.IdToToken(predictedId);
    }
    
    /// <summary>
    /// モデルをファイルに保存
    /// </summary>
    public void SaveModel(string filepath)
    {
        using (var writer = new System.IO.BinaryWriter(System.IO.File.Create(filepath)))
        {
            // メタデータ
            writer.Write(vocabSize);
            writer.Write(hiddenDim);
            writer.Write(numLayers);
            writer.Write(seqLength);
            
            // Embedding層
            writer.Write(embeddings.Data.Length);
            foreach (float val in embeddings.Data)
                writer.Write(val);
            
            // 出力層
            writer.Write(outputWeight.Data.Length);
            foreach (float val in outputWeight.Data)
                writer.Write(val);
        }
        
        Console.WriteLine($"[Model] チェックポイント保存: {filepath}");
    }
    
    /// <summary>
    /// モデルをファイルから読み込み
    /// </summary>
    public static TinyLLM LoadModel(string filepath)
    {
        using (var reader = new System.IO.BinaryReader(System.IO.File.OpenRead(filepath)))
        {
            // メタデータ
            int vs = reader.ReadInt32();
            int hd = reader.ReadInt32();
            int nl = reader.ReadInt32();
            int sl = reader.ReadInt32();
            
            var model = new TinyLLM(vs, hd, nl, sl);
            
            // Embedding層
            int embLen = reader.ReadInt32();
            for (int i = 0; i < embLen; i++)
                model.embeddings.Data[i] = reader.ReadSingle();
            
            // 出力層
            int outLen = reader.ReadInt32();
            for (int i = 0; i < outLen; i++)
                model.outputWeight.Data[i] = reader.ReadSingle();
            
            Console.WriteLine("[Model] チェックポイント読み込み完了");
            return model;
        }
    }
}
