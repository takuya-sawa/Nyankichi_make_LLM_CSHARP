using System;

/// ===================================================================
/// テンソル クラス：多次元配列の管理
/// 
/// TensorはC++のstd::vectorに相当し、ニューラルネットワークの
/// 重みと活性化値を保持します。
/// 
/// 主な機能：
/// - Shape管理（2次元行列中心）
/// - ランダム初期化
/// - ゼロ初期化
/// - 要素アクセス
/// ===================================================================
public class Tensor
{
    /// <summary>
    /// テンソルの形状：[rows, cols]
    /// </summary>
    public int[] Shape { get; private set; }
    
    /// <summary>
    /// 実際のデータ（1次元配列）
    /// </summary>
    public float[] Data { get; private set; }
    
    /// <summary>
    /// 総要素数
    /// </summary>
    public int Size => Data.Length;
    
    /// <summary>
    /// ランダムジェネレータ（共有）
    /// </summary>
    private static readonly Random rng = new Random(42);
    
    /// <summary>
    /// コンストラクタ：形状を指定してテンソルを作成
    /// </summary>
    /// <param name="shape">形状配列 [rows, cols]</param>
    public Tensor(params int[] shape)
    {
        Shape = shape;
        int size = 1;
        foreach (int dim in shape)
        {
            size *= dim;
        }
        Data = new float[size];
    }
    
    /// <summary>
    /// ランダム初期化：正規分布 N(0, 0.01)
    /// 
    /// 訓練開始時にニューロンが死なないように小さな値で初期化
    /// Xavier初期化に近い戦略
    /// </summary>
    public void RandomInit()
    {
        for (int i = 0; i < Data.Length; i++)
        {
            // Box-Muller変換で正規分布を生成
            double u1 = rng.NextDouble();
            double u2 = rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            Data[i] = (float)(z * 0.01);
        }
    }
    
    /// <summary>
    /// ゼロ初期化：すべての値を0にクリア
    /// 
    /// バイアスなどの初期化に使用
    /// </summary>
    public void Zero()
    {
        Array.Fill(Data, 0f);
    }
    
    /// <summary>
    /// 要素アクセス：行列インデックスから直線インデックスに変換
    /// </summary>
    public float Get(int row, int col)
    {
        if (Shape.Length != 2)
            throw new ArgumentException("Get requires 2D tensor");
        return Data[row * Shape[1] + col];
    }
    
    /// <summary>
    /// 要素設定：行列インデックスから直線インデックスに変換
    /// </summary>
    public void Set(int row, int col, float value)
    {
        if (Shape.Length != 2)
            throw new ArgumentException("Set requires 2D tensor");
        Data[row * Shape[1] + col] = value;
    }
    
    /// <summary>
    /// 要素アクセス：1次元テンソル用
    /// </summary>
    public float Get(int idx)
    {
        return Data[idx];
    }
    
    /// <summary>
    /// 要素設定：1次元テンソル用
    /// </summary>
    public void Set(int idx, float value)
    {
        Data[idx] = value;
    }
    
    /// <summary>
    /// テンソルをコピー
    /// </summary>
    public Tensor Clone()
    {
        var result = new Tensor(Shape);
        Array.Copy(Data, result.Data, Data.Length);
        return result;
    }
}
