use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::tokenizer::{Encoding, PostProcessor, Result};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(tag = "type", rename = "IdRemappingProcessor")]
pub struct IdRemappingProcessor {
    /// Mapping from input IDs to output IDs
    #[serde(with = "id_map_serde")]
    id_map: HashMap<u32, u32>,
}

// Custom serialization for the id_map to ensure consistent format
mod id_map_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S>(map: &HashMap<u32, u32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert the map to a vec of tuples for consistent ordering
        let mut pairs: Vec<_> = map.iter().collect();
        pairs.sort_by_key(|&(k, _)| k);
        pairs.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<u32, u32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let pairs: Vec<(u32, u32)> = Vec::deserialize(deserializer)?;
        Ok(pairs.into_iter().collect())
    }
}

impl IdRemappingProcessor {
    pub fn new(id_map: HashMap<u32, u32>) -> Self {
        Self { id_map }
    }
}

impl PostProcessor for IdRemappingProcessor {
    fn added_tokens(&self, _is_pair: bool) -> usize {
        // We're not adding any tokens, just remapping IDs
        0
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        _add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        for encoding in &mut encodings {
            // Remap the IDs in the main encoding
            let new_ids: Vec<u32> = encoding
                .get_ids()
                .iter()
                .map(|&id| self.id_map.get(&id).copied().unwrap_or(id))
                .collect();
            
            // Update the IDs in the encoding
            encoding.set_ids(new_ids);

            // Also process any overflowing encodings
            for overflow in encoding.get_overflowing_mut() {
                let new_overflow_ids: Vec<u32> = overflow
                    .get_ids()
                    .iter()
                    .map(|&id| self.id_map.get(&id).copied().unwrap_or(id))
                    .collect();
                overflow.set_ids(new_overflow_ids);
            }
        }

        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let id_map = HashMap::from([
            (1, 100),
            (2, 200),
            (3, 300),
        ]);
        let processor = IdRemappingProcessor::new(id_map);
        
        // Test serialization
        let serialized = serde_json::to_string(&processor).unwrap();
        assert_eq!(
            serialized,
            r#"{"type":"IdRemappingProcessor","id_map":[[1,100],[2,200],[3,300]]}"#
        );

        // Test deserialization
        let deserialized: IdRemappingProcessor = serde_json::from_str(&serialized).unwrap();
        assert_eq!(processor, deserialized);
    }

    #[test]
    fn test_process_encodings() {
        let id_map = HashMap::from([
            (1, 100),
            (2, 200),
        ]);
        let processor = IdRemappingProcessor::new(id_map);

        let encoding = Encoding::new(
            vec![1, 2, 3],             // ids
            vec![0, 0, 0],             // type_ids
            vec!["a", "b", "c"].into_iter().map(String::from).collect(), // tokens
            vec![None, None, None],    // words
            vec![(0, 1), (1, 2), (2, 3)], // offsets
            vec![0, 0, 0],             // special_tokens_mask
            vec![1, 1, 1],             // attention_mask
            vec![],                    // overflowing
            HashMap::new(),            // sequence_ranges
        );

        let processed = processor.process_encodings(vec![encoding], false).unwrap();
        assert_eq!(processed[0].get_ids(), &[100, 200, 3]);
    }
}
