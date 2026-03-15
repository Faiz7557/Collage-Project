<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
    public function up(): void {
        Schema::create('transaksis', function (Blueprint $table) {
            $table->id('id_transaksi');
            $table->foreignId('id_penjual')->constrained('users', 'id_user');
            $table->foreignId('id_pembeli')->constrained('users', 'id_user');
            $table->foreignId('id_pasar')->constrained('pasars', 'id_pasar');
            $table->foreignId('id_wallet')->constrained('e_wallets', 'id_wallet');
            $table->integer('jumlah');
            $table->decimal('harga_awalan', 15, 2);
            $table->decimal('harga_akhir', 15, 2);
            $table->date('tanggal');
            $table->enum('jenis_transaksi', ['jual', 'beli']);
            $table->enum('status_transaksi', ['pending', 'confirmed', 'cancelled']);
            $table->timestamps();
        });
    }
    public function down(): void { Schema::dropIfExists('transaksis'); }
};