using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

/// ===================================================================
/// TinyLLM - C# ポート版メインプログラム
/// 
/// 機能：
/// - テキストファイルからの訓練データ読み込み
/// - モデルの訓練と推論
/// - チェックポイント保存・読み込み
/// - コマンドライン引数対応（train/infer/both）
/// ===================================================================
public class Program
{
    // 訓練設定
    private const int VOCAB_SIZE = 128;
    private const int HIDDEN_DIM = 128;
    private const int NUM_LAYERS = 2;
    private const int SEQ_LENGTH = 16;
    private const int EPOCHS = 10;
    private const int STEPS_PER_EPOCH = 3;
    private const float LEARNING_RATE = 0.001f;
    
    static void Main(string[] args)
    {
        Console.WriteLine("╔════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         TinyLLM - C# ポート版                          ║");
        Console.WriteLine("║  小型言語モデル（教育用）                               ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════╝");
        Console.WriteLine();
        
        // コマンドライン引数解析
        bool runTrain = true;
        bool runInfer = true;
        
        if (args.Length > 0)
        {
            switch (args[0].ToLower())
            {
                case "train":
                    runTrain = true;
                    runInfer = false;
                    break;
                case "infer":
                    runTrain = false;
                    runInfer = true;
                    break;
                case "both":
                    runTrain = true;
                    runInfer = true;
                    break;
                default:
                    Console.WriteLine($"不明なモード: {args[0]}");
                    Console.WriteLine("使用方法: dotnet run [train|infer|both]");
                    return;
            }
        }
        
        // 訓練データの読み込み
        Console.WriteLine("[データ] 訓練データを読み込んでいます...");
        string[] trainingTexts = LoadTrainingData("data/training_data.txt");
        
        if (trainingTexts.Length == 0)
        {
            Console.WriteLine("エラー: 訓練データが見つかりません");
            return;
        }
        
        // トークナイザーの初期化
        Console.WriteLine("[トークナイザー] 語彙を構築しています...");
        var tokenizer = new SimpleTokenizer(trainingTexts);
        Console.WriteLine();
        
        // モデルの初期化または読み込み
        TinyLLM model;
        string checkpointPath = "model_checkpoint.dat";
        
        if (File.Exists(checkpointPath))
        {
            Console.WriteLine("[モデル] 既存のチェックポイントを読み込んでいます...");
            model = TinyLLM.LoadModel(checkpointPath);
            Console.WriteLine();
        }
        else
        {
            Console.WriteLine("[モデル] 新しいモデルを初期化しています...");
            model = new TinyLLM(Math.Max(tokenizer.VocabSize, VOCAB_SIZE), HIDDEN_DIM, NUM_LAYERS, SEQ_LENGTH, LEARNING_RATE);
            Console.WriteLine();
        }
        
        // 訓練フェーズ
        if (runTrain)
        {
            Console.WriteLine("╔════════════════════════════════════════════════════════╗");
            Console.WriteLine("║           訓練フェーズ開始                             ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════╝");
            Console.WriteLine();
            
            var random = new Random(42);
            
            for (int epoch = 0; epoch < EPOCHS; epoch++)
            {
                Console.WriteLine($"[訓練] エポック {epoch + 1}/{EPOCHS}");
                
                for (int step = 0; step < STEPS_PER_EPOCH; step++)
                {
                    // ランダムなテキストを選択
                    string text = trainingTexts[random.Next(trainingTexts.Length)];
                    int[] tokenIds = tokenizer.Tokenize(text);
                    
                    // 訓練ステップ：シーケンスの最後のトークン以外を入力、最後のトークンを予測目標
                    if (tokenIds.Length > 1)
                    {
                        int[] inputTokens = new int[tokenIds.Length - 1];
                        Array.Copy(tokenIds, inputTokens, inputTokens.Length);
                        int targetId = tokenIds[tokenIds.Length - 1];
                        
                        float loss = model.TrainStep(inputTokens, targetId);
                        Console.WriteLine($"  ステップ {step + 1}/{STEPS_PER_EPOCH}: Loss = {loss:F5}");
                    }
                }
                
                Console.WriteLine();
            }
            
            // チェックポイント保存
            model.SaveModel(checkpointPath);
            Console.WriteLine();
        }
        
        // 推論フェーズ
        if (runInfer)
        {
            Console.WriteLine("╔════════════════════════════════════════════════════════╗");
            Console.WriteLine("║           推論フェーズ開始                             ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════╝");
            Console.WriteLine();
            
            // テスト入力
            string[] testInputs = new[]
            {
                "I am a",
                "The cat is",
                "I like",
                "Cats are"
            };
            
            Console.WriteLine("[推論] 次のトークン予測:");
            Console.WriteLine();
            
            foreach (string input in testInputs)
            {
                int[] tokenIds = tokenizer.Tokenize(input);
                string predicted = model.Predict(tokenIds, tokenizer);
                Console.WriteLine($"  入力: \"{input}\"");
                Console.WriteLine($"  予測: \"{predicted}\"");
                Console.WriteLine();
            }
        }
        
        Console.WriteLine("処理完了！");
    }
    
    /// <summary>
    /// 訓練データをファイルから読み込み
    /// </summary>
    static string[] LoadTrainingData(string filepath)
    {
        if (!File.Exists(filepath))
        {
            Console.WriteLine($"警告: ファイルが見つかりません: {filepath}");
            return new string[0];
        }
        
        var lines = new List<string>();
        
        foreach (string line in File.ReadAllLines(filepath))
        {
            string trimmed = line.Trim();
            
            // 空行とコメント（#で始まる行）をスキップ
            if (!string.IsNullOrEmpty(trimmed) && !trimmed.StartsWith("#"))
            {
                lines.Add(trimmed);
            }
        }
        
        Console.WriteLine($"[データ] {lines.Count} 件の訓練文を読み込みました: {filepath}");
        
        return lines.ToArray();
    }
}
