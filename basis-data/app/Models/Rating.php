<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Rating extends Model
{
    use HasFactory;

    protected $fillable = [
        'id_user',
        'id_produk',
        'id_penjual',
        'nilai_rating',
        'komentar'
    ];

    public function user()
    {
        return $this->belongsTo(User::class, 'id_user', 'id_user');
    }

    public function produk()
    {
        return $this->belongsTo(ProdukBeras::class, 'id_produk');
    }

    public function penjual()
    {
        return $this->belongsTo(User::class, 'id_penjual', 'id_user');
    }
}
