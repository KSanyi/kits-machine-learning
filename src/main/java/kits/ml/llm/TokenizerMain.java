package kits.ml.llm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class TokenizerMain {

    public static void main(String[] args) throws IOException {
        
        String text = new String(Files.readAllBytes(Paths.get("input/jokai.txt")));

        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.train(text, 10000);
        
//        String example = """
//                It's the last he painted, you know,
//                Mrs. Gisburn said with pardonable pride.""";
//        List<Integer> encoded = tokenizer.encode(example);
        
        String example = "állítólag egyiptomi hieroglifok hagyományaiból kompiláltatott";
        
        List<Integer> encoded = tokenizer.encode(example);
        
        for(int tokenId : encoded) {
            System.out.println(tokenId + " -> " + tokenizer.decode(List.of(tokenId)));
        }
        
        String decoded = tokenizer.decode(encoded);
        
        System.out.println(encoded);
        System.out.println(decoded);
        
    }

}
