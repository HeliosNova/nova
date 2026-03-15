-- Fix 5a: Delete junk user facts
DELETE FROM user_facts WHERE key = 'error';
DELETE FROM user_facts WHERE value LIKE '%I like for %';

-- Fix 5b: Merge duplicate language facts (keep the one with highest id, delete the rest)
DELETE FROM user_facts
WHERE key LIKE '%language%'
  AND id NOT IN (
    SELECT MAX(id) FROM user_facts WHERE key LIKE '%language%'
  );

-- Fix 5c: Delete duplicate skills by name (keep highest times_used per name)
DELETE FROM skills
WHERE id NOT IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY LOWER(name)
            ORDER BY times_used DESC, id DESC
        ) AS rn
        FROM skills
    ) ranked
    WHERE rn = 1
);
