using System;
using System.Collections.Generic;
using System.Linq;

/// ===================================================================
/// シンプルなトークナイザー：テキスト → トークンID変換
/// 
/// 機能：
/// - 訓練データから動的に語彙を構築
/// - テキストをトークンIDの配列に変換
/// - 特別トークン対応（<PAD>, <UNK>）
/// ===================================================================
public class SimpleTokenizer
{
    /// <summary>
    /// 語彙：単語 → トークンID のマッピング
    /// </summary>
    private Dictionary<string, int> vocab = new();
    
    /// <summary>
    /// 逆マッピング：トークンID → 単語
    /// </summary>
    private Dictionary<int, string> idToWord = new();
    
    /// <summary>
    /// 語彙サイズ
    /// </summary>
    public int VocabSize => vocab.Count;
    
    /// <summary>
    /// PAD トークン ID
    /// </summary>
    private const int PAD_ID = 0;
    
    /// <summary>
    /// UNK (未知語) トークン ID
    /// </summary>
    private const int UNK_ID = 1;
    
    /// <summary>
    /// コンストラクタ：訓練テキストから語彙を構築
    /// </summary>
    /// <param name="trainingTexts">訓練用テキストの配列</param>
    public SimpleTokenizer(string[] trainingTexts)
    {
        BuildVocabulary(trainingTexts);
    }
    
    /// <summary>
    /// 語彙を構築：訓練テキストから一意の単語を抽出
    /// </summary>
    private void BuildVocabulary(string[] trainingTexts)
    {
        var uniqueWords = new HashSet<string>();
        
        // すべてのテキストから単語を抽出
        foreach (string text in trainingTexts)
        {
            string[] words = text.ToLower()
                .Split(new[] { ' ', '\t', '\n', '\r' }, 
                       StringSplitOptions.RemoveEmptyEntries);
            
            foreach (string word in words)
            {
                // 句読点を除去（簡易版）
                string cleanWord = System.Text.RegularExpressions.Regex.Replace(
                    word, @"[^\w]", "");
                
                if (!string.IsNullOrEmpty(cleanWord))
                {
                    uniqueWords.Add(cleanWord);
                }
            }
        }
        
        // 特別トークンを追加
        vocab["<pad>"] = PAD_ID;
        idToWord[PAD_ID] = "<pad>";
        
        vocab["<unk>"] = UNK_ID;
        idToWord[UNK_ID] = "<unk>";
        
        // 通常の単語を追加（ID: 2以降）
        int id = 2;
        foreach (string word in uniqueWords.OrderBy(w => w))
        {
            vocab[word] = id;
            idToWord[id] = word;
            id++;
        }
        
        Console.WriteLine($"[Tokenizer] 語彙サイズ: {VocabSize} ({uniqueWords.Count} 個の一意の単語 + 特別トークン2個)");
    }
    
    /// <summary>
    /// テキストをトークンID配列に変換
    /// </summary>
    /// <param name="text">入力テキスト</param>
    /// <returns>トークンID配列</returns>
    public int[] Tokenize(string text)
    {
        string[] words = text.ToLower()
            .Split(new[] { ' ', '\t', '\n', '\r' }, 
                   StringSplitOptions.RemoveEmptyEntries);
        
        var tokenIds = new int[words.Length];
        
        for (int i = 0; i < words.Length; i++)
        {
            // 句読点を除去
            string cleanWord = System.Text.RegularExpressions.Regex.Replace(
                words[i], @"[^\w]", "");
            
            if (!string.IsNullOrEmpty(cleanWord) && vocab.TryGetValue(cleanWord, out int id))
            {
                tokenIds[i] = id;
            }
            else
            {
                tokenIds[i] = UNK_ID; // 未知語は UNK_ID
            }
        }
        
        return tokenIds;
    }
    
    /// <summary>
    /// トークンIDを単語に変換
    /// </summary>
    public string IdToToken(int id)
    {
        return idToWord.TryGetValue(id, out string word) ? word : "<unk>";
    }
    
    /// <summary>
    /// トークンID配列を単語配列に変換
    /// </summary>
    public string[] DeTokenize(int[] tokenIds)
    {
        return tokenIds.Select(id => IdToToken(id)).ToArray();
    }
}
